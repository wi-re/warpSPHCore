#pragma once
#include "execution/fieldTraversal.h"
#include "execution/atomic.h"
#include "kernels/adjoint.h"
#include "torch/validation.h"
namespace sph_native {
template <class T> struct GradientBuffers {
    T *pointers[OUTPUT];
    int64_t mask;
    SPH_HD void add(int slot, int64_t index, T value) const {
        if (mask & (int64_t(1) << slot))
            add_atomic(pointers[slot] + index, value);
    }
};
template <class T, int D> struct PairBars {
    T bw = T(0), bg[D] = {}, bx[D] = {}, br = T(0), bh = T(0), bm = T(0), bqr = T(0), brr = T(0),
      lap_bar = T(0);
};
template <class T, int D>
SPH_HD void weighted_vjp(const RowArgs<T, D> &a, int64_t i, int64_t j, int c, T bar, bool lap,
                         PairBars<T, D> &b, const GradientBuffers<T> &out) {
    T fi = a.qvalue[i * a.fields + c], fj = a.rvalue[j * a.fields + c], m = a.mass[j],
      rho = a.density[j];
    if (a.scheme == 2 && !lap) {
        T qi = a.qdensity[i];
        out.add(QVALUE, i * a.fields + c, bar * m / qi);
        out.add(RVALUE, j * a.fields + c, bar * m * qi / (rho * rho));
        b.bm += bar * (fi / qi + fj * qi / (rho * rho));
        b.bqr += bar * m * (-fi / (qi * qi) + fj / (rho * rho));
        b.brr -= bar * T(2) * m * qi * fj / (rho * rho * rho);
    } else {
        T combination = lap || a.scheme == 3 ? fj - fi : (a.scheme == 1 ? fj : fj + fi);
        T coefficient = m / rho, dm = T(1) / rho, dr = -m / (rho * rho);
        if (lap && a.scheme == 2) {
            coefficient = m * a.qdensity[i] / (rho * rho);
            dm = a.qdensity[i] / (rho * rho);
            dr = -T(2) * coefficient / rho;
            b.bqr += bar * combination * m / (rho * rho);
        }
        out.add(RVALUE, j * a.fields + c, bar * coefficient);
        if (lap || a.scheme != 1)
            out.add(QVALUE, i * a.fields + c, (lap || a.scheme == 3 ? -bar : bar) * coefficient);
        b.bm += bar * combination * dm;
        b.brr += bar * combination * dr;
    }
}
template <class T, int D, class Pair> struct AdjointRow {
    RowArgs<T, D> a;
    const T *cotangent;
    GradientBuffers<T> out;
    SPH_HD void operator()(int64_t i) const {
        if (!query_allowed(a.qkind[i], a.direction))
            return;
        int64_t start = a.offsets[i];
        for (int32_t edge = 0; edge < a.counts[i]; ++edge) {
            int64_t j = a.neighbors[start + edge];
            if (!reference_allowed(a.rkind[j], a.direction))
                continue;
            T dx[D], w, g, r = a.displacement(i, j, dx);
            pair_kernel(a, i, j, r, w, g);
            PairBars<T, D> b;
            Pair::backward(a, i, j, dx, r, w, g, cotangent + i * a.outputs, b, out);
            T bhi = T(0), bhj = T(0);
            if (b.lap_bar != T(0)) {
                if (a.support == 15) {
                    laplacian_vjp<T, D>(r, a.qh[i], a.kernel, b.lap_bar / T(2), b.br, bhi);
                    laplacian_vjp<T, D>(r, a.rh[j], a.kernel, b.lap_bar / T(2), b.br, bhj);
                } else
                    laplacian_vjp<T, D>(r, a.radius(i, j), a.kernel, b.lap_bar, b.br, b.bh);
            }
            if (a.support == 14 || a.support == 15) {
                T bg[D];
                for (int d = 0; d < D; ++d)
                    bg[d] = b.bg[d] / T(2);
                kernel_vjp<T, D>(dx, a.qh[i], a.kernel, b.bw / T(2), bg, b.bx, bhi);
                kernel_vjp<T, D>(dx, a.rh[j], a.kernel, b.bw / T(2), bg, b.bx, bhj);
            } else {
                T bh = T(0);
                kernel_vjp<T, D>(dx, a.radius(i, j), a.kernel, b.bw, b.bg, b.bx, bh);
                b.bh += bh;
            }
            if (a.support == 11)
                bhi += b.bh;
            else if (a.support == 12)
                bhj += b.bh;
            else if (a.support == 13) {
                bhi += b.bh / T(2);
                bhj += b.bh / T(2);
            } else if (a.qh[i] >= a.rh[j])
                bhi += b.bh;
            else
                bhj += b.bh;
            out.add(QH, i, bhi);
            out.add(RH, j, bhj);
            out.add(MASS, j, b.bm);
            out.add(DENSITY, j, b.brr);
            out.add(QDENSITY, i, b.bqr);
            for (int d = 0; d < D; ++d) {
                T bar = b.bx[d] + (r > T(0) ? b.br * dx[d] / r : T(0));
                out.add(QX, i * D + d, bar);
                out.add(RX, j * D + d, -bar);
                if (a.periodic[d]) {
                    T winding =
                        round((a.qx[i * D + d] - a.rx[j * D + d] - dx[d]) / (a.hi[d] - a.lo[d]));
                    out.add(LO, d, bar * winding);
                    out.add(HI, d, -bar * winding);
                }
            }
        }
    }
};
template <class Pair> struct AdjointRows {
    const Tensors &gradients;
    const Tensor &cotangent;
    int64_t mask;
    template <class T, int D, class Launcher>
    void operator()(int64_t n, RowArgs<T, D> a, const Launcher &launch) const {
        GradientBuffers<T> buffers{};
        buffers.mask = mask;
        for (int slot = 0; slot < OUTPUT; ++slot)
            buffers.pointers[slot] = gradients[slot].mutable_data_ptr<T>();
        launch(n, AdjointRow<T, D, typename Pair::template type<T, D>>{
                      a, cotangent.const_data_ptr<T>(), buffers});
    }
};
inline Tensors gradient_buffers(const Tensors &t, int64_t mask) {
    Tensors gradients;
    for (int slot = 0; slot < OUTPUT; ++slot) {
        if (mask & (int64_t(1) << slot))
            gradients.push_back(stable::new_zeros(t[slot], t[slot].sizes()));
        else
            gradients.push_back(stable::new_empty(t[QX], {0}));
    }
    return gradients;
}
} // namespace sph_native
