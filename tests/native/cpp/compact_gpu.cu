#include "gpu_storage.h"
#include "hash_cases.h"
#include "execution/compactTraversal.h"
#include <iostream>
#include <limits>
using namespace hash_checks;

template <class T, int D>
static Problem upload(Storage &storage, const Fixture<T, D> &fixture, Support support) {
    auto problem = fixture.problem(support);
    auto copy = [&](auto &values) { return storage.upload(values.data(), int64_t(values.size())); };
    problem.query = {problem.query.count, copy(fixture.query), copy(fixture.query_support)};
    problem.reference = {problem.reference.count, copy(fixture.reference),
                         copy(fixture.reference_support)};
    problem.domain = {copy(fixture.lower), copy(fixture.upper), copy(fixture.periodic)};
    return problem;
}
static void compare(Storage &storage, const Context &context, Problem problem,
                    const CompactHash &index, const Expected &expected) {
    const auto queries = problem.query.count;
    Buffer counts{queries ? storage.allocate(size_t(queries) * sizeof(int32_t)) : nullptr, queries};
    Buffer offsets{queries ? storage.allocate(size_t(queries) * sizeof(int64_t)) : nullptr,
                   queries};
    radius_count(context, problem, index, counts);
    const auto total = radius_scan(context, queries, {counts.data, counts.size}, offsets);
    std::vector<int32_t> got_counts(queries);
    std::vector<int64_t> got_offsets(queries);
    if (queries) {
        storage.download(got_counts, counts);
        storage.download(got_offsets, offsets);
    }
    require(got_counts == expected.counts && got_offsets == expected.offsets &&
                total == int64_t(expected.indices.size()),
            "Compact GPU counts/offsets differ from all-pairs oracle");
    constexpr int64_t canary = -37891;
    std::vector<int64_t> indices(size_t(total) + 3, canary);
    const auto allocation = storage.upload(indices.data(), int64_t(indices.size()));
    Buffer output{const_cast<void *>(allocation.data), total};
    problem.adjacency = {{offsets.data, queries}, {counts.data, queries}, {output.data, total}};
    radius_collect(context, problem, index, output);
    storage.download(indices, output);
    require(std::vector<int64_t>(indices.begin(), indices.begin() + total) == expected.indices,
            "Compact GPU original IDs differ from all-pairs oracle");
    for (size_t i = size_t(total); i < indices.size(); ++i)
        require(indices[i] == canary, "GPU collect exceeded capacity");
}
template <class T, int D> static void matrix() {
    for (int mask = 0; mask < (1 << D); ++mask)
        for (int scenario = 0; scenario < 3; ++scenario) {
            Storage storage;
            Context context{device_kind, 0, storage.stream};
            Fixture<T, D> fixture(37, 53, mask, scenario);
            for (int policy = 11; policy <= 15; ++policy) {
                const auto problem = upload(storage, fixture, Support(policy));
                const auto expected = oracle(fixture, Support(policy));
                for (int64_t buckets : {int64_t(1), int64_t(7), int64_t(0)}) {
                    auto index = build_compact_hash(context, problem, {buckets});
                    statistics(index, problem.reference.count, buckets);
                    compare(storage, context, problem, index, expected);
                }
            }
        }
}
static void reuse_and_errors() {
    Storage storage;
    Context context{device_kind, 0, storage.stream};
    Fixture<double, 2> fixture(17, 21, 1, 0);
    auto problem = upload(storage, fixture, Support::Gather);
    auto index = build_compact_hash(context, problem, {1});
    compare(storage, context, problem, index, oracle(fixture, Support::Gather));
    for (auto &position : fixture.query)
        position += .37;
    problem.query.positions = storage.upload(fixture.query.data(), fixture.query.size());
    compare(storage, context, problem, index, oracle(fixture, Support::Gather));
    const Buffer counts{storage.allocate(17 * sizeof(int32_t)), 17};
    rejects([&] {
        radius_count(context, problem, index,
                     {const_cast<int64_t *>(index.view().reference_keys), 17});
    });
    const double saved_support = fixture.query_support[0];
    const auto stats = index.statistics();
    fixture.query_support[0] = std::nextafter(stats.radius_bound, stats.cell_width);
    problem.query.supports =
        storage.upload(fixture.query_support.data(), fixture.query_support.size());
    rejects([&] { radius_count(context, problem, index, counts); });
    fixture.query_support[0] = saved_support;
    problem.query.supports =
        storage.upload(fixture.query_support.data(), fixture.query_support.size());
    fixture.reference[0] += .51;
    problem.reference.positions =
        storage.upload(fixture.reference.data(), fixture.reference.size());
    context.validation = Validation::Metadata;
    rejects([&] { radius_count(context, problem, index, counts); });
    fixture.reference[0] -= .51;
    problem.reference.positions =
        storage.upload(fixture.reference.data(), fixture.reference.size());
    fixture.query_support[0] = 100;
    problem.query.supports =
        storage.upload(fixture.query_support.data(), fixture.query_support.size());
    rejects([&] { radius_count(context, problem, index, counts); });
    fixture.query_support[0] = .01;
    problem.query.supports =
        storage.upload(fixture.query_support.data(), fixture.query_support.size());
    fixture.lower[0] -= .1;
    problem.domain.min = storage.upload(fixture.lower.data(), 2);
    rejects([&] { radius_count(context, problem, index, counts); });
    fixture.lower[0] += .1;
    problem.domain.min = storage.upload(fixture.lower.data(), 2);
    fixture.reference_support[0] = -1;
    problem.reference.supports =
        storage.upload(fixture.reference_support.data(), fixture.reference_support.size());
    rejects([&] { build_compact_hash(context, problem); });
    fixture.reference_support[0] = .01;
    problem.reference.supports =
        storage.upload(fixture.reference_support.data(), fixture.reference_support.size());
    fixture.query[0] = std::numeric_limits<double>::quiet_NaN();
    problem.query.positions = storage.upload(fixture.query.data(), fixture.query.size());
    rejects([&] { build_compact_hash(context, problem); });
}
static void boundaries_and_empty() {
    Storage storage;
    Context context{device_kind, 0, storage.stream};
    for (int periodic = 0; periodic < 2; ++periodic) {
        Fixture<double, 1> fixture(3, 5, periodic, 0);
        fixture.query = {0., 4., -.5};
        fixture.reference = {.25, -.25, 1., -.5, .5};
        fixture.query_support = {.25, .25, 0.};
        fixture.reference_support.assign(5, .25);
        const auto problem = upload(storage, fixture, Support::Gather);
        auto index = build_compact_hash(context, problem, {1});
        compare(storage, context, problem, index, oracle(fixture, Support::Gather));
    }
    for (auto sizes : {std::pair{0, 5}, std::pair{7, 0}, std::pair{0, 0}}) {
        const auto [queries, references] = sizes;
        Fixture<double, 3> fixture(queries, references, 7, 0);
        // Empty uploads need no device allocation or transfer.
        auto problem = fixture.problem(Support::Gather);
        auto copy = [&](const auto &values) -> ConstBuffer {
            return values.empty() ? ConstBuffer{}
                                  : storage.upload(values.data(), int64_t(values.size()));
        };
        problem.query = {queries, copy(fixture.query), copy(fixture.query_support)};
        problem.reference = {references, copy(fixture.reference), copy(fixture.reference_support)};
        problem.domain = {copy(fixture.lower), copy(fixture.upper), copy(fixture.periodic)};
        for (int policy = 11; policy <= 15; ++policy) {
            problem.properties.support = Support(policy);
            auto index = build_compact_hash(context, problem);
            compare(storage, context, problem, index, oracle(fixture, Support(policy)));
        }
    }
}
__global__ void candidate_counts(const double *query, compact::View index, int64_t queries,
                                 int64_t *counts) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= queries)
        return;
    compact::CellTraversal<double, 3> traversal(query + i * 3, index);
    compact::Cell cell{};
    int64_t count = 0;
    while (traversal.next(cell))
        count += cell.end - cell.begin;
    counts[i] = count;
}
static void larger_and_sparse() {
    Storage storage;
    Context context{device_kind, 0, storage.stream};
    for (int64_t references : {int64_t(1027), int64_t(65537)}) {
        Fixture<double, 3> fixture(3, references, 5, 0);
        const auto problem = upload(storage, fixture, Support::SuperSymmetric);
        auto index = build_compact_hash(context, problem, {7});
        compare(storage, context, problem, index, oracle(fixture, Support::SuperSymmetric));
    }
    Fixture<double, 3> fixture(37, 53, 7, 0);
    const auto problem = upload(storage, fixture, Support::Gather);
    auto index = build_compact_hash(context, problem, {1});
    const Buffer counts{storage.allocate(37 * sizeof(int64_t)), 37};
    candidate_counts<<<1, 128, 0, storage.stream>>>(
        static_cast<const double *>(problem.query.positions.data), index.view(), 37,
        static_cast<int64_t *>(counts.data));
    check(GPU(GetLastError)());
    std::vector<int64_t> candidates(37);
    storage.download(candidates, counts);
    int64_t total = 0;
    for (auto count : candidates)
        total += count;
    require(total < problem.query.count * problem.reference.count,
            "Localized GPU hash traversal scanned every reference");
    fixture.query_support.assign(37, 0);
    fixture.reference_support.assign(53, 0);
    const auto zero = upload(storage, fixture, Support::Gather);
    auto sparse = build_compact_hash(context, zero);
    require(sparse.statistics().grid_cells[0] > 100000, "Sparse GPU grid unexpectedly dense");
    compare(storage, context, zero, sparse, oracle(fixture, Support::Gather));
}

template <class T> static void tiny_distances() {
    Storage storage;
    Context context{device_kind, 0, storage.stream};
    Fixture<T, 1> fixture(1, 2, 0, 0);
    const T extent = sizeof(T) == sizeof(float) ? T(1e-25) : T(1e-200);
    const T support = sizeof(T) == sizeof(float) ? T(1e-40) : T(1e-300);
    fixture.lower = {0};
    fixture.upper = {extent};
    fixture.query = {0};
    fixture.reference = {0, extent};
    fixture.query_support = {support};
    fixture.reference_support = {support, support};
    const auto problem = upload(storage, fixture, Support::Gather);
    auto index = build_compact_hash(context, problem);
    compare(storage, context, problem, index, Expected{{2}, {0}, {0, 1}});
}

int main() {
    int devices = 0;
    if (GPU(GetDeviceCount)(&devices) != GPU(Success) || !devices)
        return 77;
    try {
        matrix<float, 1>();
        matrix<float, 2>();
        matrix<float, 3>();
        matrix<double, 1>();
        matrix<double, 2>();
        matrix<double, 3>();
        reuse_and_errors();
        boundaries_and_empty();
        larger_and_sparse();
        tiny_distances<float>();
        tiny_distances<double>();
        std::cout << "Compact GPU hash: 1260 oracle cases, collisions, periodic masks, reuse and "
                     "bounds passed on a caller stream\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
