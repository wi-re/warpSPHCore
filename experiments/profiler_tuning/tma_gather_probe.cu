// Compile-only availability probe. This kernel is never executed: a real TMA
// operation also needs a valid tensor map and initialized transaction barrier.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void tma_gather4_probe(void *shared_destination, const void *tensor_map,
                                uint64_t *shared_barrier, int column) {
    const auto destination = unsigned(__cvta_generic_to_shared(shared_destination));
    const auto barrier = unsigned(__cvta_generic_to_shared(shared_barrier));
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
                 "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                 :: "r"(destination), "l"(tensor_map), "r"(column), "r"(0), "r"(1), "r"(2), "r"(3),
                    "r"(barrier) : "memory");
}
