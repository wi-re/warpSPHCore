#pragma once
#include "api.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace hash_checks {
using namespace sph_native;
template <class Container> ConstBuffer input(const Container &values) {
    return {values.data(), int64_t(values.size())};
}
template <class Container> Buffer output(Container &values) {
    return {values.data(), int64_t(values.size())};
}
inline void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void rejects(F call) {
    try {
        call();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("Invalid compact hash request was accepted");
}
template <class T, int D> struct Fixture {
    std::vector<T> query, reference, query_support, reference_support;
    std::array<T, D> lower{}, upper{};
    std::array<bool, D> periodic{};
    Fixture(int64_t queries, int64_t references, int mask, int scenario)
        : query(queries * D), reference(references * D), query_support(queries),
          reference_support(references) {
        std::mt19937 generator(781 + D + mask + scenario);
        std::uniform_real_distribution<double> random(0, 1);
        for (int axis = 0; axis < D; ++axis) {
            lower[axis] = -1;
            upper[axis] = T(1 + axis);
            periodic[axis] = (mask & (1 << axis)) != 0;
        }
        auto positions = [&](auto &values) {
            for (size_t i = 0; i < values.size(); ++i) {
                const int axis = i % D;
                const T length = upper[axis] - lower[axis];
                values[i] = T(lower[axis] + (random(generator) * 1.4 - .2) * length);
                if (periodic[axis])
                    values[i] += T((int(i % 9) - 4) * length);
            }
        };
        positions(query);
        positions(reference);
        const double radius = scenario == 0 ? .13 : (scenario == 1 ? .95 : 9.0);
        for (auto &value : query_support)
            value = T(radius * (.2 + .8 * random(generator)));
        for (auto &value : reference_support)
            value = T(radius * (.2 + .8 * random(generator)));
    }
    Problem problem(Support support) const {
        Problem problem{};
        problem.precision = sizeof(T) == sizeof(float) ? Precision::Float32 : Precision::Float64;
        problem.dimension = D;
        problem.query = {int64_t(query_support.size()), input(query), input(query_support)};
        problem.reference = {int64_t(reference_support.size()), input(reference),
                             input(reference_support)};
        problem.domain = {input(lower), input(upper), input(periodic)};
        problem.properties.support = support;
        return problem;
    }
};
struct Expected {
    std::vector<int32_t> counts;
    std::vector<int64_t> offsets, indices;
};
// Independent all-pairs oracle: long-double remainder gives the nearest image
// directly. It does not use the native distance, grid, hash or support helpers.
template <class T, int D> Expected oracle(const Fixture<T, D> &fixture, Support support) {
    Expected expected;
    for (size_t i = 0; i < fixture.query_support.size(); ++i) {
        expected.offsets.push_back(int64_t(expected.indices.size()));
        int32_t count = 0;
        for (size_t j = 0; j < fixture.reference_support.size(); ++j) {
            long double squared = 0;
            for (int axis = 0; axis < D; ++axis) {
                long double delta =
                    (long double)fixture.query[i * D + axis] - fixture.reference[j * D + axis];
                if (fixture.periodic[axis]) {
                    const long double length =
                        (long double)fixture.upper[axis] - fixture.lower[axis];
                    delta = std::remainder(delta, length);
                }
                squared += delta * delta;
            }
            const T left = fixture.query_support[i], right = fixture.reference_support[j];
            const T radius = support == Support::Gather          ? left
                             : support == Support::Scatter       ? right
                             : support == Support::MeanSymmetric ? (left + right) / T(2)
                                                                 : (left > right ? left : right);
            if (std::sqrt(squared) < radius) {
                expected.indices.push_back(int64_t(j));
                ++count;
            }
        }
        expected.counts.push_back(count);
    }
    return expected;
}
inline void statistics(const CompactHash &index, int64_t references, int64_t buckets) {
    const auto stats = index.statistics();
    require(stats.references == references, "Incorrect indexed reference count");
    require(stats.occupied_cells <= references && (stats.occupied_cells > 0 || !references),
            "Invalid occupied-cell count");
    require(stats.hash_buckets == (buckets ? buckets : (references ? references : 1)),
            "Incorrect bucket count");
}
} // namespace hash_checks
