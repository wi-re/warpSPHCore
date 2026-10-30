#pragma once
#include "execution/adjointTraversal.h"
namespace sph_native {
template <class T, int D> struct LaplacianAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        T h = a.radius(i, j), r2 = r * r;
        if (a.laplacian_mode == 1) {
            T P;
            if (a.support == 15)
                P = (evaluate_kernel<T, D>(dx, a.qh[i], a.kernel).laplacian +
                     evaluate_kernel<T, D>(dx, a.rh[j], a.kernel).laplacian) /
                    T(2);
            else
                P = evaluate_kernel<T, D>(dx, h, a.kernel).laplacian;
            T bp = T(0);
            for (int c = 0; c < a.fields; ++c) {
                weighted_vjp(a, i, j, c, bar[c] * P, true, b, out);
                bp += bar[c] * field_weight(a, i, j, c, true);
            }
            // Support-specific Naive Laplacian bars are handled by the row below.
            b.lap_bar = bp;
        } else if (r > T(0)) {
            T A = r + T(1e-8) * h;
            if (a.laplacian_mode == 3) {
                T n[D], bn[D] = {}, F = g * r2 / (A * A), bF = T(0), bA = T(0);
                for (int d = 0; d < D; ++d)
                    n[d] = dx[d] / A;
                for (int base = 0; base < a.fields; base += D) {
                    T projection = T(0), bproj = T(0);
                    for (int d = 0; d < D; ++d)
                        projection += field_weight(a, i, j, base + d, true) * n[d];
                    for (int d = 0; d < D; ++d) {
                        T z = bar[base + d], f = field_weight(a, i, j, base + d, true);
                        weighted_vjp(a, i, j, base + d, z * F, true, b, out);
                        bF += z * (f - T(D + 2) * projection * n[d]);
                        bproj -= z * T(D + 2) * n[d] * F;
                        bn[d] -= z * T(D + 2) * projection * F;
                    }
                    for (int d = 0; d < D; ++d) {
                        weighted_vjp(a, i, j, base + d, bproj * n[d], true, b, out);
                        bn[d] += bproj * field_weight(a, i, j, base + d, true);
                    }
                }
                T ng = T(0);
                for (int d = 0; d < D; ++d) {
                    T G = g * dx[d];
                    ng += n[d] * G;
                    bn[d] += bF * G / A;
                    b.bg[d] += bF * n[d] / A;
                }
                bA -= bF * ng / (A * A);
                for (int d = 0; d < D; ++d) {
                    b.bx[d] += bn[d] / A;
                    bA -= bn[d] * dx[d] / (A * A);
                }
                b.br += bA;
                b.bh += T(1e-8) * bA;
            } else {
                T e2 = a.laplacian_mode == 4 ? T(1e-12) : T(1e-8), B = r + e2 * h,
                  P = -T(2) * g * r2 / (A * B), bp = T(0);
                for (int c = 0; c < a.fields; ++c) {
                    weighted_vjp(a, i, j, c, bar[c] * P, true, b, out);
                    bp += bar[c] * field_weight(a, i, j, c, true);
                }
                for (int d = 0; d < D; ++d) {
                    b.bg[d] -= bp * T(2) * dx[d] / (A * B);
                    b.bx[d] -= bp * T(2) * g * dx[d] / (A * B);
                }
                b.br -= bp * P * (T(1) / A + T(1) / B);
                b.bh -= bp * P * (T(1e-8) / A + e2 / B);
            }
        }
    }
};
struct LaplacianAdjointPolicy {
    template <class T, int D> using type = LaplacianAdjointPair<T, D>;
};
} // namespace sph_native
