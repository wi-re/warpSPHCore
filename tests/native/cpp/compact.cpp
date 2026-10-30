#include "hash_cases.h"
#include "execution/compactTraversal.h"
#include <iostream>
#include <limits>
#include <utility>
using namespace hash_checks;

static void compare(const Context &context, Problem problem, const CompactHash &index,
                    const Expected &expected) {
    constexpr int64_t canary = -37891;
    std::vector<int32_t> counts(expected.counts.size());
    std::vector<int64_t> offsets(expected.offsets.size());
    radius_count(context, problem, index, output(counts));
    require(counts == expected.counts, "Compact CPU counts differ from all-pairs oracle");
    const auto total = radius_scan(context, problem.query.count, input(counts), output(offsets));
    require(offsets == expected.offsets && total == int64_t(expected.indices.size()),
            "Compact CPU scan mismatch");
    std::vector<int64_t> indices(size_t(total) + 3, canary);
    problem.adjacency = {input(offsets), input(counts), input(indices)};
    radius_collect(context, problem, index, {indices.data(), total});
    require(std::vector<int64_t>(indices.begin(), indices.begin() + total) == expected.indices,
            "Compact CPU original IDs differ from all-pairs oracle");
    for (size_t i = size_t(total); i < indices.size(); ++i)
        require(indices[i] == canary, "Collect exceeded capacity");
}
template <class T, int D> static void matrix(const Context &context) {
    for (int mask = 0; mask < (1 << D); ++mask)
        for (int scenario = 0; scenario < 3; ++scenario) {
            Fixture<T, D> fixture(37, 53, mask, scenario);
            for (int policy = 11; policy <= 15; ++policy) {
                const auto problem = fixture.problem(Support(policy));
                const auto expected = oracle(fixture, Support(policy));
                for (int64_t buckets : {int64_t(1), int64_t(7), int64_t(0)}) {
                    auto index = build_compact_hash(context, problem, {buckets});
                    statistics(index, problem.reference.count, buckets);
                    compare(context, problem, index, expected);
                }
            }
        }
}
static void reuse_and_errors(Context context) {
    Fixture<double, 2> fixture(17, 21, 1, 0);
    auto problem = fixture.problem(Support::Gather);
    auto index = build_compact_hash(context, problem, {1});
    auto moved = std::move(index);
    rejects([&] { index.statistics(); });
    compare(context, problem, moved, oracle(fixture, Support::Gather));
    for (auto &position : fixture.query)
        position += .37;
    compare(context, fixture.problem(Support::Gather), moved, oracle(fixture, Support::Gather));
    std::vector<int32_t> counts(17);
    rejects([&] {
        radius_count(context, problem, moved,
                     {const_cast<int64_t *>(moved.view().reference_keys), 17});
    });
    const double saved_support = fixture.query_support[0];
    const auto stats = moved.statistics();
    fixture.query_support[0] = std::nextafter(stats.radius_bound, stats.cell_width);
    rejects(
        [&] { radius_count(context, fixture.problem(Support::Gather), moved, output(counts)); });
    fixture.query_support[0] = saved_support;
    const double saved = fixture.reference[0];
    fixture.reference[0] += .51;
    context.validation = Validation::Metadata; // stale-cell checks also apply here
    rejects(
        [&] { radius_count(context, fixture.problem(Support::Gather), moved, output(counts)); });
    fixture.reference[0] = saved;
    fixture.query_support[0] = 100;
    rejects(
        [&] { radius_count(context, fixture.problem(Support::Gather), moved, output(counts)); });
    fixture.query_support[0] = .01;
    fixture.lower[0] -= .1;
    rejects(
        [&] { radius_count(context, fixture.problem(Support::Gather), moved, output(counts)); });
    fixture.lower[0] += .1;
    fixture.reference_support[0] = -1;
    rejects([&] { build_compact_hash(context, fixture.problem(Support::Gather)); });
    fixture.reference_support[0] = .01;
    rejects([&] { build_compact_hash(context, fixture.problem(Support::Gather), {-1}); });
    problem = fixture.problem(Support::Gather);
    problem.query.positions.size = 1;
    rejects([&] { build_compact_hash(context, problem); });
    fixture.query[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { build_compact_hash(context, fixture.problem(Support::Gather)); });
    fixture.query[0] = 0;
    fixture.upper[0] = fixture.lower[0];
    rejects([&] { build_compact_hash(context, fixture.problem(Support::Gather)); });
}
static void boundaries_and_empty(const Context &context) {
    for (int periodic = 0; periodic < 2; ++periodic) {
        Fixture<double, 1> fixture(3, 5, periodic, 0);
        fixture.query = {0., 4., -.5};
        fixture.reference = {.25, -.25, 1., -.5, .5};
        fixture.query_support = {.25, .25, 0.};
        fixture.reference_support.assign(5, .25);
        const auto problem = fixture.problem(Support::Gather);
        auto index = build_compact_hash(context, problem, {1});
        compare(context, problem, index, oracle(fixture, Support::Gather));
    }
    for (auto sizes : {std::pair{0, 5}, std::pair{7, 0}, std::pair{0, 0}}) {
        const auto [queries, references] = sizes;
        Fixture<double, 3> fixture(queries, references, 7, 0);
        for (int policy = 11; policy <= 15; ++policy) {
            const auto problem = fixture.problem(Support(policy));
            auto index = build_compact_hash(context, problem);
            compare(context, problem, index, oracle(fixture, Support(policy)));
        }
    }
}
static void larger_and_sparse(const Context &context) {
    for (int64_t references : {int64_t(1027), int64_t(65537)}) {
        Fixture<double, 3> fixture(3, references, 5, 0);
        const auto problem = fixture.problem(Support::SuperSymmetric);
        auto index = build_compact_hash(context, problem, {7});
        compare(context, problem, index, oracle(fixture, Support::SuperSymmetric));
    }
    Fixture<double, 3> fixture(37, 53, 7, 0);
    const auto problem = fixture.problem(Support::Gather);
    auto index = build_compact_hash(context, problem, {1});
    int64_t candidates = 0;
    const auto view = index.view();
    for (int64_t i = 0; i < problem.query.count; ++i) {
        compact::CellTraversal<double, 3> traversal(fixture.query.data() + 3 * i, view);
        compact::Cell cell{};
        while (traversal.next(cell))
            candidates += cell.end - cell.begin;
    }
    require(candidates < problem.query.count * problem.reference.count,
            "Localized hash traversal scanned every reference");
    fixture.query_support.assign(37, 0);
    fixture.reference_support.assign(53, 0);
    auto sparse = build_compact_hash(context, fixture.problem(Support::Gather));
    const auto stats = sparse.statistics();
    require(stats.grid_cells[0] > 100000 && stats.occupied_cells <= 53,
            "Sparse index should not allocate its Cartesian grid volume");
    compare(context, fixture.problem(Support::Gather), sparse, oracle(fixture, Support::Gather));
}

template <class T> static void tiny_distances(const Context &context) {
    Fixture<T, 1> fixture(1, 2, 0, 0);
    const T extent = sizeof(T) == sizeof(float) ? T(1e-25) : T(1e-200);
    const T support = sizeof(T) == sizeof(float) ? T(1e-40) : T(1e-300);
    fixture.lower = {0};
    fixture.upper = {extent};
    fixture.query = {0};
    fixture.reference = {0, extent};
    fixture.query_support = {support};
    fixture.reference_support = {support, support};
    const auto problem = fixture.problem(Support::Gather);
    auto index = build_compact_hash(context, problem);
    // Both squared distances underflow to zero in the existing scalar
    // predicate. Grid pruning must preserve its accepted pairs.
    compare(context, problem, index, Expected{{2}, {0}, {0, 1}});
}

int main() {
    try {
        Context context{};
        context.cpu_threads = 4;
        matrix<float, 1>(context);
        matrix<float, 2>(context);
        matrix<float, 3>(context);
        matrix<double, 1>(context);
        matrix<double, 2>(context);
        matrix<double, 3>(context);
        reuse_and_errors(context);
        boundaries_and_empty(context);
        larger_and_sparse(context);
        tiny_distances<float>(context);
        tiny_distances<double>(context);
        std::cout << "Compact CPU hash: 1260 oracle cases, collisions, periodic masks, reuse and "
                     "bounds passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
