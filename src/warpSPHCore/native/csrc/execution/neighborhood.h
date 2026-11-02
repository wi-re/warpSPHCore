#pragma once

#include "config.h"
#include <cstdint>

namespace sph_native::neighborhood {

// Edge ordinals are optional metadata. A cell/cluster traversal can supply -1
// when there is no retained particle-edge array to address.
struct Neighbor {
    int64_t reference;
    int64_t edge;
};

template <int Size> struct NeighborBatch {
    Neighbor entries[Size];
    bool valid[Size];
};

// Offsets can be a typed pointer or a borrowed accessor such as IndexView.
// Resolve neighbor-ID width at launch; read offset width once per query.
template <class Index, class Offsets = const int64_t*> struct Csr {
    Offsets offsets;
    const int32_t* counts;
    const Index* indices;

    template <class Visit>
    SPH_HD void each(int64_t query, int lane, int lanes, Visit visit) const {
        const auto begin = offsets[query];
        for (int32_t local = lane; local < counts[query]; local += lanes)
            visit(Neighbor{int64_t(indices[begin + local]), int64_t(begin + local)});
    }

    template <int Size, class Visit>
    SPH_HD void each_batch(int64_t query, int lane, int lanes, Visit visit) const {
        const auto begin = offsets[query];
        const int32_t count = counts[query];
        for (int64_t base = lane; base < count; base += lanes * Size) {
            NeighborBatch<Size> batch{};
#pragma unroll
            for (int u = 0; u < Size; ++u) {
                const auto local = base + u * lanes;
                batch.valid[u] = local < count;
                if (batch.valid[u])
                    batch.entries[u] = {int64_t(indices[begin + local]), begin + local};
            }
            visit(batch);
        }
    }
};

// A query can visit several contiguous source ranges, including whole cells or
// clusters. It is deliberately not restricted to an explicit particle-ID list.
struct Ranges {
    const int64_t* offsets;
    const int32_t* counts;
    const int32_t* begins;
    const int32_t* ends;

    template <class Visit>
    SPH_HD void each(int64_t query, int lane, int lanes, Visit visit) const {
        for (int32_t range = 0; range < counts[query]; ++range) {
            const auto at = offsets[query] + range;
            for (int64_t j = int64_t(begins[at]) + lane; j < ends[at]; j += lanes)
                visit(Neighbor{j, -1});
        }
    }
};

struct LocalReduction {
    template <class T> SPH_HD void sum(T&) const {}
    template <class T> SPH_HD void maximum(T&) const {}
    template <class T> SPH_HD void minimum(T&) const {}
    template <class Selection> SPH_HD void winner(Selection&) const {}
};

template <class T, int Components> struct Sums {
    T values[Components] = {};

    SPH_HD void combine(const Sums& other) {
        for (int c = 0; c < Components; ++c) values[c] += other.values[c];
    }

    template <class Reduction> SPH_HD void reduce(const Reduction& reduction) {
        for (int c = 0; c < Components; ++c)
            reduction.sum(values[c]);
    }
};

template <class T, int Components> struct CountedSums {
    T values[Components] = {};
    int32_t count = 0;

    SPH_HD void combine(const CountedSums& other) {
        for (int c = 0; c < Components; ++c) values[c] += other.values[c];
        count += other.count;
    }

    template <class Reduction> SPH_HD void reduce(const Reduction& reduction) {
        for (int c = 0; c < Components; ++c)
            reduction.sum(values[c]);
        reduction.sum(count);
    }
};

template <class T> struct Maximum {
    T value = T(0);
    int32_t index = -1;

    SPH_HD void combine(const Maximum& other) {
        if (other.index >= 0) consider(other.value, other.index);
    }

    SPH_HD void consider(T candidate, int32_t candidate_index) {
        if (index < 0 || candidate > value ||
            (candidate == value && candidate_index < index)) {
            value = candidate;
            index = candidate_index;
        }
    }

    template <class Reduction> SPH_HD void reduce(const Reduction& reduction) {
        reduction.winner(*this);
    }
};

// Operation owns preparation, interaction arithmetic, the accumulator and
// finalization. Loader owns reference records. Neither knows the ID encoding.
template <class Operation, class Query, class Loader, class Accumulator>
struct Accumulate {
    const Operation& operation;
    const Query& query;
    const Loader& loader;
    Accumulator& accumulator;

    SPH_HD void operator()(Neighbor neighbor) const {
        if (loader.allowed(neighbor.reference)) {
            const auto reference = loader.load(neighbor.reference);
            operation.interact(query, reference, neighbor, accumulator);
        }
    }
};

template <class Operation, class Traversal, class Loader> struct Row {
    Operation operation;
    Traversal traversal;
    Loader loader;

    template <class Reduction>
    SPH_HD void run(int64_t i, int lane, int lanes, const Reduction& reduction) const {
        const auto query = operation.prepare(i);
        auto accumulator = operation.identity(query);
        if (operation.allowed(i))
            traversal.each(i, lane, lanes,
                Accumulate<Operation, decltype(query), Loader, decltype(accumulator)>{
                    operation, query, loader, accumulator});
        accumulator.reduce(reduction);
        if (lane == 0)
            operation.finish(query, accumulator);
    }

    SPH_HD void operator()(int64_t i) const {
        run(i, 0, 1, LocalReduction{});
    }
};

} // namespace sph_native::neighborhood
