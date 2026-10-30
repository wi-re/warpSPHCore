// A minimal CPU workflow: create particle data, build a neighbourhood, compute
// density, then use that density to evaluate the gradient of a scalar field.
//
// Run sph_example for the primal workflow. Add --mass-gradient to also show an
// explicit derivative of sum(density) with respect to the particle masses.
#include "api.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace sph = sph_native;

// These helpers borrow existing storage; they do not allocate or copy data.
// Buffer capacities are element counts, not byte counts. The arrays/vectors
// must remain alive while the library uses their pointers.
template <class Container> sph::ConstBuffer as_input_buffer(const Container &values) {
    return {values.data(), static_cast<int64_t>(values.size())};
}

template <class Container> sph::Buffer as_output_buffer(Container &values) {
    return {values.data(), static_cast<int64_t>(values.size())};
}

void print_density_mass_sensitivity(const sph::Context &context,
                                    const sph::Problem &density_problem);

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--mass-gradient")) {
        std::cerr << "Usage: sph_example [--mass-gradient]\n";
        return 1;
    }
    const bool show_mass_gradient = argc == 2;

    try {
        // 1. Own the input data in ordinary C++ containers.
        //
        // This is a two-particle, one-dimensional example. Positions are stored
        // as flattened [particle_count, dimension] arrays. Supports and masses
        // have one entry per particle; scalar field values have shape [N].
        constexpr std::size_t particle_count = 2;
        const std::array<double, particle_count> positions = {-0.2, 0.2};
        const std::array<double, particle_count> support_radii = {1.0, 1.0};
        const std::array<double, particle_count> masses = {1.0, 2.0};
        const std::array<double, particle_count> scalar_values = {0.0, 1.0};
        const std::array<int32_t, particle_count> particle_kinds = {0, 0};

        // Domain arrays have one entry per spatial dimension. This domain is
        // periodic on [-1, 1), so distances use the nearest periodic image.
        const std::array<double, 1> domain_min = {-1.0};
        const std::array<double, 1> domain_max = {1.0};
        const std::array<bool, 1> periodic_axes = {true};

        sph::Particles particles;
        particles.count = particle_count;
        particles.positions = as_input_buffer(positions);
        particles.supports = as_input_buffer(support_radii);
        particles.masses = as_input_buffer(masses);
        particles.kinds = as_input_buffer(particle_kinds);
        // Density is an output of the first operation, so it is not supplied yet.

        // 2. Describe the calculation using views of those containers.
        //
        // Query particles receive outputs. Reference particles contribute to
        // their neighbour sums. Here both roles refer to the same population.
        sph::Problem problem;
        problem.precision = sph::Precision::Float64;
        problem.dimension = 1;
        problem.field_rank = 0; // scalar field; use 1 for vectors or 2 for matrices
        problem.query = particles;
        problem.reference = particles;
        problem.domain.min = as_input_buffer(domain_min);
        problem.domain.max = as_input_buffer(domain_max);
        problem.domain.periodic = as_input_buffer(periodic_axes);
        problem.properties.kernel = sph::Kernel::Wendland2;
        problem.properties.support = sph::Support::Gather;

        sph::Context context;
        context.device = sph::Device::CPU;
        context.cpu_threads = 2;
        context.validation = sph::Validation::Contents;
        // CPU calls complete before returning. Contents validation checks the
        // borrowed data as well as buffer sizes before launching an operation.

        // 3. Build a CSR neighbour list: count, scan, allocate, collect.
        //
        // counts[i] is the number of neighbours of query i. offsets[i] is the
        // start of that query's row in the flat neighbour_indices array.
        // Self neighbours are included. For this fixture the rows are [0, 1]
        // and [0, 1], with counts [2, 2] and offsets [0, 2].
        // Build the occupied-cell index once and reuse it for both passes.
        // References crossing cell boundaries or a larger search radius require
        // rebuilding it; changing query positions is supported within this bound.
        auto search_index = sph::build_compact_hash(context, problem);
        std::vector<int32_t> neighbour_counts(particle_count);
        sph::radius_count(context, problem, search_index, as_output_buffer(neighbour_counts));

        std::vector<int64_t> neighbour_offsets(particle_count);
        const int64_t edge_count =
            sph::radius_scan(context, particles.count, as_input_buffer(neighbour_counts),
                             as_output_buffer(neighbour_offsets));

        std::vector<int64_t> neighbour_indices(static_cast<std::size_t>(edge_count));
        problem.adjacency.counts = as_input_buffer(neighbour_counts);
        problem.adjacency.offsets = as_input_buffer(neighbour_offsets);
        problem.adjacency.indices = as_input_buffer(neighbour_indices);
        sph::radius_collect(context, problem, search_index, as_output_buffer(neighbour_indices));

        // 4. Compute density: density[i] = sum_j mass[j] * W(i, j).
        //
        // The application owns output storage too. output_elements() gives the
        // required capacity for the selected operation and field shape.
        problem.properties.operation = sph::Operation::Density;
        const sph::Problem density_problem = problem;
        std::vector<double> density(
            static_cast<std::size_t>(sph::output_elements(density_problem)));
        sph::forward(context, density_problem, as_output_buffer(density));

        // 5. Compute a scalar-field gradient using the newly computed density.
        //
        // Difference mode weights (field[j] - field[i]) by mass[j]/density[j]
        // and the kernel gradient. Its output has shape [N, dimension].
        // The same CSR can be reused because geometry and supports are unchanged.
        sph::Problem gradient_problem = density_problem;
        gradient_problem.properties.operation = sph::Operation::Gradient;
        gradient_problem.properties.scheme = sph::Scheme::Difference;
        gradient_problem.query.densities = as_input_buffer(density);
        gradient_problem.reference.densities = as_input_buffer(density);
        gradient_problem.query_values = as_input_buffer(scalar_values);
        gradient_problem.reference_values = as_input_buffer(scalar_values);

        std::vector<double> scalar_gradient(
            static_cast<std::size_t>(sph::output_elements(gradient_problem)));
        sph::forward(context, gradient_problem, as_output_buffer(scalar_gradient));

        std::cout << "Neighbour entries: " << edge_count << " (including self neighbours)\n";
        for (std::size_t particle = 0; particle < particle_count; ++particle) {
            std::cout << "Particle " << particle << ": density = " << density[particle]
                      << ", scalar-field gradient = " << scalar_gradient[particle] << '\n';
        }

        // Optional: differentiate just the density operation. All its input
        // containers and CSR buffers are still alive and unchanged here.
        if (show_mass_gradient) {
            print_density_mass_sensitivity(context, density_problem);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Standalone example failed: " << error.what() << '\n';
        return 1;
    }
}

void print_density_mass_sensitivity(const sph::Context &context,
                                    const sph::Problem &density_problem) {
    // Choose the scalar objective L = sum_i density[i]. Its derivative with
    // respect to each density output is 1, so the incoming cotangent is all ones.
    std::vector<double> density_cotangent(
        static_cast<std::size_t>(sph::output_elements(density_problem)), 1.0);
    std::vector<double> mass_gradient(static_cast<std::size_t>(density_problem.reference.count));

    // Request only dL/dmass. MASS names the reference-mass input slot. By default
    // adjoint() clears the requested gradient buffer before accumulating into it.
    sph::Adjoints requested_gradients;
    requested_gradients.requested = sph::gradient_bit(sph::MASS);
    requested_gradients.slots[sph::MASS] = as_output_buffer(mass_gradient);
    sph::adjoint(context, density_problem, as_input_buffer(density_cotangent), requested_gradients);

    // This is an explicit VJP of density. It does not differentiate through the
    // subsequent gradient operation or construct an automatic differentiation graph.
    std::cout << "Mass sensitivity of L = sum(density):\n";
    for (std::size_t particle = 0; particle < mass_gradient.size(); ++particle) {
        std::cout << "  dL/dmass[" << particle << "] = " << mass_gradient[particle] << '\n';
    }
}
