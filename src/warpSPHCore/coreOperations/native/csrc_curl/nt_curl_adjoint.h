#pragma once
#include "execution/adjointTraversal.h"
#include "nt_curl_contract.h"
#include "nt_curl_kernel.h"
namespace sph_native {
template <class T, int D> struct CurlAdjointPair {
    SPH_HD static void backward(const RowArgs<T, D> &a, int64_t i, int64_t j, const T (&dx)[D], T r,
                                T w, T g, const T *bar, PairBars<T, D> &b,
                                const GradientBuffers<T> &out) {
        T field_bars[D*D]={},gradient[D]={};
        if constexpr(D>1)curl_kernel_gradient(a,i,j,dx,r,gradient);
        if constexpr (D == 2) {
            for (int c = 0; c < a.outputs; ++c) {
                field_bars[a.outputs+c]=bar[c]*gradient[0];
                field_bars[c]=-bar[c]*gradient[1];
                b.bg[0] += bar[c] * field_weight(a, i, j, a.outputs + c);
                b.bg[1] -= bar[c] * field_weight(a, i, j, c);
            }
        }
        if constexpr (D == 3) {
            int S = a.fields / 3;
            for (int c = 0; c < S; ++c) {
                T u[3] = {bar[c], bar[S + c], bar[2 * S + c]},
                  f[3] = {field_weight(a, i, j, c), field_weight(a, i, j, S + c),
                          field_weight(a, i, j, 2 * S + c)};
                T bf[3],bg[3];
                curl_pair_bars(u,f,gradient,bf,bg);
                for(int d=0;d<3;++d) {
                    field_bars[d*S+c]=bf[d];
                    b.bg[d]+=bg[d];
                }
            }
        }
        if constexpr(D>1) {
            if(a.scheme==2) {
                for(int c=0;c<a.fields;++c)weighted_vjp<T,D,true>(a,i,j,c,field_bars[c],false,b,out);
            } else {
                const T mass=a.mass[j],rho=a.density[j],volume=mass/rho;
                T volume_bar=0;
                for(int c=0;c<a.fields;++c) {
                    const T fi=a.qvalue[i*a.fields+c],fj=a.rvalue[j*a.fields+c];
                    const T combination=a.scheme==1 ? fj:a.scheme==3 ? fj-fi:fj+fi;
                    // Reverse the vector coefficient first, then its scalar
                    // mass/density quotient, in the reference's order.
                    volume_bar+=warp_compat::rounded_scalar_term(field_bars[c]*combination);
                    out.add(RVALUE,j*a.fields+c,field_bars[c]*volume);
                    if(a.scheme!=1)out.add(QVALUE,i*a.fields+c,(a.scheme==3 ? -field_bars[c]:field_bars[c])*volume);
                }
                b.bm+=volume_bar/rho;
                b.brr-=volume_bar*mass/(rho*rho);
            }
        }
    }
};
struct CurlAdjointPolicy {
    template <class T, int D> using type = CurlAdjointPair<T, D>;
};
} // namespace sph_native
