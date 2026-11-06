#pragma once
#include "execution/fieldTraversal.h"
#include "execution/atomic.h"
#include "kernels/adjoint.h"
#include "kernels/warp_derivatives.h"
#include "dataTypes/slots.h"
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
template <class T, int D, bool ReferenceScalarProducts=false>
SPH_HD void weighted_vjp(const RowArgs<T, D> &a, int64_t i, int64_t j, int c, T bar, bool lap,
                         PairBars<T, D> &b, const GradientBuffers<T> &out) {
    auto term=[](T value) {if constexpr(ReferenceScalarProducts)return warp_compat::rounded_scalar_term(value);else return value;};
    T fi = a.qvalue[i * a.fields + c], fj = a.rvalue[j * a.fields + c], m = a.mass[j],
      rho = a.density[j];
    if (a.scheme == 2 && !lap) {
        T qi = a.qdensity[i];
        out.add(QVALUE, i * a.fields + c, bar * m / qi);
        out.add(RVALUE, j * a.fields + c, bar * m * qi / (rho * rho));
        b.bm += term(bar * (fi / qi + fj * qi / (rho * rho)));
        b.bqr += term(bar * m * (-fi / (qi * qi) + fj / (rho * rho)));
        b.brr -= term(bar * T(2) * m * qi * fj / (rho * rho * rho));
    } else {
        T combination = lap || a.scheme == 3 ? fj - fi : (a.scheme == 1 ? fj : fj + fi);
        T coefficient = m / rho, dm = T(1) / rho, dr = -m / (rho * rho);
        if (lap && a.scheme == 2) {
            coefficient = m * a.qdensity[i] / (rho * rho);
            dm = a.qdensity[i] / (rho * rho);
            dr = -T(2) * coefficient / rho;
            b.bqr += term(bar * combination * m / (rho * rho));
        }
        out.add(RVALUE, j * a.fields + c, bar * coefficient);
        if (lap || a.scheme != 1)
            out.add(QVALUE, i * a.fields + c, (lap || a.scheme == 3 ? -bar : bar) * coefficient);
        b.bm += term(bar * combination * dm);
        b.brr += term(bar * combination * dr);
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
                    laplacian_pullback<T, D>(dx,r, a.qh[i], a.kernel, b.lap_bar / T(2), b.bx, bhi);
                    laplacian_pullback<T, D>(dx,r, a.rh[j], a.kernel, b.lap_bar / T(2), b.bx, bhj);
                } else
                    laplacian_pullback<T, D>(dx,r, a.radius(i, j), a.kernel, b.lap_bar, b.bx, b.bh);
            }
            if (a.operation == 1 || a.operation == 6) {
                auto value_pullback = [&](T h, T bar, T& support_bar) {
                    T value = T(0), gradient[D] = {}, dh = T(0);
                    warp_compat::value_derivatives<T,D>(dx, r, h, a.kernel, value, gradient, dh);
                    for (int d = 0; d < D; ++d) b.bx[d] += bar * gradient[d];
                    support_bar += bar * dh;
                };
                if (a.support == 14 || a.support == 15) {
                    value_pullback(a.qh[i], b.bw / T(2), bhi);
                    value_pullback(a.rh[j], b.bw / T(2), bhj);
                } else value_pullback(a.radius(i, j), b.bw, b.bh);
            } else if(force_operation(a.operation)) {
                if(a.support==14||a.support==15) {
                    T half[D];for(int d=0;d<D;++d)half[d]=b.bg[d]/T(2);
                    warp_compat::force_pullback<T,D>(dx,r,a.qh[i],a.kernel,T(0),half,b.bx,bhi);
                    warp_compat::force_pullback<T,D>(dx,r,a.rh[j],a.kernel,T(0),half,b.bx,bhj);
                } else warp_compat::force_pullback<T,D>(dx,r,a.radius(i,j),a.kernel,T(0),b.bg,b.bx,b.bh);
            } else if (a.support == 14 || a.support == 15) {
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
            } else if ((a.operation == 1 || force_operation(a.operation) || a.operation == 6) ? a.qh[i] > a.rh[j] : a.qh[i] >= a.rh[j])
                bhi += b.bh;
            else
                bhj += b.bh;
            out.add(QH, i, bhi);
            out.add(RH, j, bhj);
            out.add(MASS, j, b.bm);
            out.add(DENSITY, j, b.brr);
            out.add(QDENSITY, i, b.bqr);
            for (int d = 0; d < D; ++d) {
                T radial = r>T(0) ? (a.operation==5 ? T(2)*dx[d]*((T(.5)/r)*b.br):b.br*dx[d]/r):T(0);
                T bar = b.bx[d] + radial;
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
} // namespace sph_native
