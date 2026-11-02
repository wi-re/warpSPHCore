#pragma once

#include "execution/rowNeighborhood.h"
#include "execution/neighborhood.cuh"

namespace sph_native::neighborhood {

template <int Batch, class Operation, class Traversal, class Loader, class Launch>
void launch_cooperative_row(int64_t n, Operation operation, Traversal traversal,
                            Loader loader, const Launch& launch) {
    if constexpr (Batch == 1) {
        using Kernel = Row<Operation, Traversal, Loader>;
        launch(n, CooperativeRow<Launch::width, Kernel>{{operation, traversal, loader}});
    } else {
        using Kernel = BatchedRow<Batch, Operation, Traversal, Loader>;
        launch(n, CooperativeRow<Launch::width, Kernel>{{operation, traversal, loader}});
    }
}

template <int Batch = 1, class Operation, class Loader, class T, int D, class Launch>
void launch_cooperative_csr(int64_t n, Operation operation, Loader loader,
                            const RowArgs<T, D>& a, const Launch& launch) {
    if (a.neighbors.wide)
        launch_cooperative_row<Batch>(n, operation, csr_view<int64_t>(a), loader, launch);
    else
        launch_cooperative_row<Batch>(n, operation, csr_view<int32_t>(a), loader, launch);
}

template <class Loader> struct PrepareRecords {
    Loader loader;
    void* destination;

    __device__ void operator()(int64_t j, int) const {
        using Record = decltype(loader.load(j));
        static_cast<Record*>(destination)[j] = loader.load(j);
    }
};

template <class Loader, class Launch>
void prepare_records(int64_t references, Loader loader, void* destination, const Launch& launch) {
    CooperativeGpuLauncher<1>{launch.stream}(references, PrepareRecords<Loader>{loader, destination});
}

template <template <class, int, int> class Policy, class T, Weight Kind,
          bool Packed = false, int Batch = 1>
struct CooperativeSphRows {
    T* packed = nullptr;

    template <int D, int Fields, class Launch>
    void fields(int64_t n, RowArgs<T, D> a, const Launch& launch) const {
        using Operation = Policy<T, D, Fields>;
        constexpr int stored = Operation::field_count;
        auto input = sph_fields<T, D, stored>(a);
        if constexpr (!Operation::uses_query_values) input.query_values = nullptr;
        Operation operation{};
        operation.fields = input;
        using Raw = ReferenceLoader<T, D, stored, Kind>;
        if constexpr (Packed) {
            using Prepared = ReferenceLoader<T, D, stored, Kind, true>;
            prepare_records(a.references, Raw{input}, packed, launch);
            launch_cooperative_csr<Batch>(n, operation, Prepared{input, packed}, a, launch);
        } else {
            launch_cooperative_csr<Batch>(n, operation, Raw{input}, a, launch);
        }
    }

    template <int D, class Launch>
    void operator()(int64_t n, RowArgs<T, D> a, const Launch& launch) const {
        if (a.fields == 1) fields<D, 1>(n, a, launch);
        else if (a.fields == D) fields<D, D>(n, a, launch);
        else fields<D, D * D>(n, a, launch);
    }
};

} // namespace sph_native::neighborhood
