#include "api.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#if defined(SPH_NATIVE_HIP)
#include <hip/hip_runtime.h>
#define GPU(name) hip##name
using Stream = hipStream_t;
constexpr auto device_kind = sph_native::Device::HIP;
#else
#include <cuda_runtime.h>
#define GPU(name) cuda##name
using Stream = cudaStream_t;
constexpr auto device_kind = sph_native::Device::CUDA;
#endif
using namespace sph_native;
static void check(GPU(Error_t) e) {
    if (e != GPU(Success))
        throw std::runtime_error(GPU(GetErrorString)(e));
}
struct Storage {
    std::vector<void *> allocations;
    Stream stream{};
    Storage() { check(GPU(StreamCreate)(&stream)); }
    ~Storage() {
        GPU(StreamSynchronize)(stream);
        for (auto p : allocations)
            GPU(Free)(p);
        GPU(StreamDestroy)(stream);
    }
    void *allocate(size_t bytes) {
        void *p = nullptr;
        check(GPU(Malloc)(&p, bytes));
        allocations.push_back(p);
        return p;
    }
    template <class T> ConstBuffer upload(const T *p, int64_t n) {
        auto out = allocate(size_t(n) * sizeof(T));
        check(GPU(MemcpyAsync)(out, p, size_t(n) * sizeof(T), GPU(MemcpyHostToDevice), stream));
        return {out, n};
    }
    template <class T> void download(std::vector<T> &dest, Buffer src) {
        check(GPU(MemcpyAsync)(dest.data(), src.data, dest.size() * sizeof(T),
                               GPU(MemcpyDeviceToHost), stream));
        check(GPU(StreamSynchronize)(stream));
    }
};
template <class T, size_t N> ConstBuffer view(const T (&a)[N]) { return {a, int64_t(N)}; }
template <class T> ConstBuffer view(const std::vector<T> &a) {
    return {a.data(), int64_t(a.size())};
}
template <class T> Buffer writable(std::vector<T> &a) { return {a.data(), int64_t(a.size())}; }
static void compare(const std::vector<double> &a, const std::vector<double> &b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > 1e-10 * (1 + std::abs(b[i])))
            throw std::runtime_error("Standalone GPU mismatch");
}
int main() {
    int devices = 0;
    if (GPU(GetDeviceCount)(&devices) != GPU(Success) || !devices) {
        std::cout << "No GPU available\n";
        return 77;
    }
    try {
        Storage storage;
        Context cpu{}, gpu{};
        gpu.device = device_kind;
        gpu.stream = storage.stream;
        const double x[] = {-.2, 0., .2, .1}, h[] = {1., 1.}, m[] = {1., 2.}, rho[] = {1., 1.},
                     f[] = {0., 1., 1., 0.}, lo[] = {-1., -1.}, hi[] = {1., 1.};
        const int32_t kinds[] = {0, 0};
        const bool periodic[] = {true, false};
        Problem p{};
        p.dimension = 2;
        p.field_rank = 1;
        p.query = p.reference = {2, view(x), view(h), view(m), view(rho), view(kinds)};
        p.domain = {view(lo), view(hi), view(periodic)};
        p.query_values = p.reference_values = view(f);
        Problem g = p;
        g.query = g.reference = {2,
                                 storage.upload(x, 4),
                                 storage.upload(h, 2),
                                 storage.upload(m, 2),
                                 storage.upload(rho, 2),
                                 storage.upload(kinds, 2)};
        g.domain = {storage.upload(lo, 2), storage.upload(hi, 2), storage.upload(periodic, 2)};
        g.query_values = g.reference_values = storage.upload(f, 4);
        std::vector<int32_t> counts(2);
        radius_count(cpu, p, writable(counts));
        std::vector<int64_t> offsets(2);
        int64_t edges = radius_scan(cpu, 2, view(counts), writable(offsets));
        std::vector<int64_t> ids(edges);
        p.adjacency = {view(offsets), view(counts), view(ids)};
        radius_collect(cpu, p, writable(ids));
        Buffer gc{storage.allocate(2 * sizeof(int32_t)), 2},
            go{storage.allocate(2 * sizeof(int64_t)), 2},
            gi{storage.allocate(edges * sizeof(int64_t)), edges};
        radius_count(gpu, g, gc);
        if (radius_scan(gpu, 2, {gc.data, gc.size}, go) != edges)
            throw std::runtime_error("CSR total mismatch");
        g.adjacency = {{go.data, go.size}, {gc.data, gc.size}, {gi.data, gi.size}};
        radius_collect(gpu, g, gi);
        std::vector<int64_t> gotids(edges);
        storage.download(gotids, gi);
        if (gotids != ids)
            throw std::runtime_error("CSR index mismatch");
        for (int op = 1; op <= 7; ++op) {
            p.properties.operation = g.properties.operation = Operation(op);
            int64_t n = output_elements(p);
            std::vector<double> expected(n), actual(n), cot(n, 1.), bx(4), bm(2), gotbx(4),
                gotbm(2);
            Buffer out{storage.allocate(n * sizeof(double)), n};
            forward(cpu, p, writable(expected));
            forward(gpu, g, out);
            storage.download(actual, out);
            compare(actual, expected);
            Adjoints a{}, b{};
            a.requested = b.requested = gradient_bit(QX) | gradient_bit(RX) | gradient_bit(MASS);
            a.slots[QX] = a.slots[RX] = writable(bx);
            a.slots[MASS] = writable(bm);
            b.slots[QX] = b.slots[RX] = {storage.allocate(4 * sizeof(double)), 4};
            b.slots[MASS] = {storage.allocate(2 * sizeof(double)), 2};
            adjoint(cpu, p, view(cot), a);
            adjoint(gpu, g, storage.upload(cot.data(), n), b);
            storage.download(gotbx, b.slots[QX]);
            storage.download(gotbm, b.slots[MASS]);
            compare(gotbx, bx);
            compare(gotbm, bm);
        }
        std::cout << "Standalone GPU: all seven primals, explicit VJPs and CSR passed on a caller "
                     "stream; no Torch/Python\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
