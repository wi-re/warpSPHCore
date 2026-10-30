// Regression checks for aliased query/reference adjoints and accumulation.
#include "api.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace sph_native;
template <class T, size_t N> ConstBuffer view(const T (&a)[N]) { return {a, int64_t(N)}; }
template <class T> ConstBuffer view(const std::vector<T> &a) {
    return {a.data(), int64_t(a.size())};
}
template <class T> Buffer writable(std::vector<T> &a) { return {a.data(), int64_t(a.size())}; }
static void require(bool b) {
    if (!b)
        throw std::runtime_error("Standalone result check failed");
}
int main() {
    try {
        const double x[] = {-.2, .2}, h[] = {1., 1.}, mass[] = {1., 2.}, rho[] = {1., 1.},
                     field[] = {0., 1.}, lo[] = {-1.}, hi[] = {1.};
        const int32_t kinds[] = {0, 0};
        const bool periodic[] = {true};
        Problem p{};
        p.dimension = 1;
        p.query = p.reference = {2, view(x), view(h), view(mass), view(rho), view(kinds)};
        p.domain = {view(lo), view(hi), view(periodic)};
        p.query_values = p.reference_values = view(field);
        Context ctx{};
        ctx.cpu_threads = 2;
        std::vector<int32_t> counts(2);
        radius_count(ctx, p, writable(counts));
        std::vector<int64_t> offsets(2);
        int64_t edges = radius_scan(ctx, 2, view(counts), writable(offsets));
        std::vector<int64_t> neighbors(edges);
        p.adjacency = {view(offsets), view(counts), view(neighbors)};
        radius_collect(ctx, p, writable(neighbors));
        require(edges == 4 && counts[0] == 2 && neighbors[3] == 1);
        std::vector<double> density(2), gradient(2), cot(2, 1.), bx(2), bh(2), bm(2);
        forward(ctx, p, writable(density));
        require(density[0] > 0 && density[1] > 0);
        Adjoints bars{};
        bars.requested = gradient_bit(QX) | gradient_bit(RX) | gradient_bit(QH) | gradient_bit(RH) |
                         gradient_bit(MASS);
        bars.slots[QX] = bars.slots[RX] = writable(bx);
        bars.slots[QH] = bars.slots[RH] = writable(bh);
        bars.slots[MASS] = writable(bm);
        adjoint(ctx, p, view(cot), bars);
        require(std::isfinite(bx[0]) && std::abs(bx[0] + bx[1]) < 1e-12 && bm[0] > 0);
        auto original = bx;
        bars.accumulate = true;
        adjoint(ctx, p, view(cot), bars);
        require(std::abs(bx[0] - 2 * original[0]) < 1e-12);
        p.properties.operation = Operation::Gradient;
        forward(ctx, p, writable(gradient));
        require(gradient[0] > 0 && gradient[1] > 0);
        std::cout << "Standalone CPU: CSR, density, gradient and explicit aliased VJP passed; no "
                     "Torch/Python\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
