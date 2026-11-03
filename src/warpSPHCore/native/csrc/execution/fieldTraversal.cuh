#pragma once
#include "execution/cooperative.cuh"
#include "execution/fieldTraversal.h"
#include "math/queryGeometry.h"
namespace sph_native {
template <int D, int Fields, int Operation> SPH_HD constexpr int field_outputs() {
    if constexpr (Operation == 2)
        return Fields * D;
    if constexpr (Operation == 3)
        return Fields / D;
    if constexpr (Operation == 4)
        return D == 1 ? 1 : (D == 2 ? Fields / D : Fields);
    if constexpr (Operation == 7)
        return D * D;
    return Fields;
}

template <class T, int D, int Fields, int Operation, int Lanes, class Pair>
struct CooperativeFieldRow {
    RowArgs<T, D> input;
    __device__ void operator()(int64_t i, int lane) const {
        constexpr int outputs = field_outputs<D, Fields, Operation>();
        RowArgs<T, D> a = input;
        a.fields = Fields;
        a.outputs = outputs;
        T sum[D * D * D] = {};
        if (query_allowed(a.qkind[i], a.direction)) {
            const QueryGeometry<T, D> geometry(a.qx + i * D, a.lo, a.hi, a.periodic);
            const bool prepared = a.kernel == 0 && a.support == 11;
            const Wendland2Primal<T, D> kernel(a.qh[i]);
            const int64_t start = a.offsets[i];
            for (int32_t edge = lane; edge < a.counts[i]; edge += Lanes) {
                const int64_t j = a.neighbors[start + edge];
                if (!reference_allowed(a.rkind[j], a.direction))
                    continue;
                T dx[D], w, g;
                const T r = geometry.distance(a.rx + j * D, dx);
                if (prepared)
                    kernel.evaluate(r, w, g);
                else
                    pair_kernel(a, i, j, r, w, g);
                Pair::accumulate(a, i, j, dx, r, w, g, sum);
            }
        }
#pragma unroll
        for (int c = 0; c < outputs; ++c) {
            const T total = group_sum<Lanes>(sum[c]);
            if (lane == 0)
                a.output[i * outputs + c] = total;
        }
    }
};

template <template <class, int> class Pair, int Operation> struct CooperativeFieldRows {
    template <class T, int D, int Fields, class Launch>
    void fields(int64_t n, RowArgs<T, D> a, const Launch &launch) const {
        launch(n, CooperativeFieldRow<T, D, Fields, Operation, Launch::width, Pair<T, D>>{a});
    }
    template <class T, int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch &launch) const {
        if (a.fields == 1)
            fields<T, D, 1>(n, a, launch);
        else if (a.fields == D)
            fields<T, D, D>(n, a, launch);
        else
            fields<T, D, D * D>(n, a, launch);
    }
};
} // namespace sph_native
