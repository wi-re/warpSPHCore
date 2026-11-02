#pragma once
#include "nt_preparation_rows.h"
#include "execution/rowNeighborhood.cuh"

namespace sph_native::crk {
struct CooperativePreparationExecutor {
    static constexpr bool device = true;
    template <class Operation, class Loader, class T, int D, class Launch>
    void operator()(int64_t n, Operation op, Loader loader, const RowArgs<T, D>& a,
                    const Launch& launch) const {
        neighborhood::launch_cooperative_csr(n, op, loader, a, launch);
    }
};
} // namespace sph_native::crk
