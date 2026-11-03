#pragma once
#include "execution/dispatch.h"
#include "math/directional.h"
#include "math/queryGeometry.h"
#include "util/directionality.h"
#include "kernels/kernel.h"
#include "kernels/adjoint.h"
#include "coreOperations/native/csrc_correction/nt_correction_torch.h"

namespace sph_native::differential {
constexpr int Slots = OUTPUT + CORRECTION_SLOTS;
constexpr int correction(int slot) { return OUTPUT + slot; }
constexpr int VOLUME = correction(REFERENCE_VOLUME), A = correction(CRK_A), B = correction(CRK_B);
constexpr int GA = correction(CRK_GRAD_A), GB = correction(CRK_GRAD_B);
constexpr int QO = correction(QUERY_OMEGA), RO = correction(REFERENCE_OMEGA), R = correction(RENORMALIZATION);
constexpr int MaxFields = 9, MaxOutputs = 52;
// Preparation kinds share the same directed pair interface as the core.
constexpr int Volume = 8, Moments = 9, Consistency = 10;

template <class T> struct Arrays {
    const T *values[Slots] = {}, *tangents[Slots] = {};
    T* bars[Slots] = {};
    SPH_HD T get(int slot, int64_t at, T fallback = T(0)) const {
        return values[slot] ? values[slot][at] : fallback;
    }
    template <class S> SPH_HD S read(int slot, int64_t at, T fallback = T(0)) const {
        if constexpr (std::is_same_v<S, T>) return get(slot, at, fallback);
        else return {get(slot, at, fallback), tangents[slot] ? tangents[slot][at] : T(0)};
    }
};

template <class S, int D> struct Pair {
    S dx[D], distance, hi, hj, h;
    S raw_w, raw_g[D], lap, w, gradient[D], before_R[D];
    S coefficient_A, coefficient_B[D], grad_A[D], grad_B[D*D], matrix[D*D], polynomial;
    S volume, mass, rhoi, rhoj, query_omega, reference_omega;
    S fi[MaxFields], fj[MaxFields], weighted[MaxFields];
    bool crk, renormalized;
};

template <class T, int D, class S>
SPH_HD void kernel(const S (&dx)[D], S h, int name, S& w, S (&gradient)[D], S& lap) {
    T x[D]; for (int d=0; d<D; ++d) x[d]=ScalarValue<S>::get(dx[d]);
    const T radius=ScalarValue<S>::get(h);
    T r2=0; for (int d=0; d<D; ++d) r2+=x[d]*x[d];
    const T r=scalar_sqrt(r2);
    T weight, radial; kernel_pair_value<T,D>(r,radius,name,weight,radial);
    const auto e=evaluate_kernel<T,D>(x,radius,name);
    if constexpr (std::is_same_v<T,S>) {
        w=weight; lap=e.laplacian;
        for (int d=0; d<D; ++d) gradient[d]=radial*x[d];
    } else {
        T dw=0, dg[D]={}, dr=0;
        bool moving=h.tangent!=0;
        for (int d=0; d<D; ++d) moving=moving || dx[d].tangent!=0;
        // In coefficient-only directions even singular kernels need no
        // geometric Hessian. Avoid multiplying an undefined Hessian by zero.
        if (moving) {
            dw=e.dh*h.tangent;
            for (int d=0; d<D; ++d) {
                dw+=e.gradient[d]*dx[d].tangent;
                dg[d]=e.gradient_dh[d]*h.tangent;
                for (int c=0; c<D; ++c) dg[d]+=e.hessian[d*D+c]*dx[c].tangent;
                if (r>0) dr+=x[d]*dx[d].tangent/r;
            }
        }
        T br=0,bh=0;
        if (moving) laplacian_vjp<T,D>(r,radius,name,T(1),br,bh);
        w={weight,dw}; lap={e.laplacian,br*dr+bh*h.tangent};
        for (int d=0; d<D; ++d) gradient[d]={radial*x[d],dg[d]};
    }
}

template <class T, int D, class S>
SPH_HD Pair<S,D> prepare(const RowArgs<T,D>& args, const Arrays<T>& arrays,
                         int64_t i, int64_t j, bool consistent) {
    Pair<S,D> p{};
    auto read=[&](int slot,int64_t at,T fallback=T(0)) { return arrays.template read<S>(slot,at,fallback); };
    T dx[D];
    const T distance=QueryGeometry<T,D>(args.qx+i*D,args.lo,args.hi,args.periodic).distance(args.rx+j*D,dx);
    T dr=0;
    for (int d=0; d<D; ++d) {
        if constexpr (std::is_same_v<T,S>) p.dx[d]=dx[d];
        else {
            const T winding=args.periodic[d] ? T(round((args.qx[i*D+d]-args.rx[j*D+d]-dx[d])/(args.hi[d]-args.lo[d]))) : T(0);
            const auto q=read(QX,i*D+d),r=read(RX,j*D+d),lo=read(LO,d),hi=read(HI,d);
            const T direction=q.tangent-r.tangent+winding*(lo.tangent-hi.tangent);
            p.dx[d]={dx[d],direction};
            if (distance>0) dr+=dx[d]*direction/distance;
        }
    }
    if constexpr (std::is_same_v<T,S>) p.distance=distance;
    else p.distance={distance,dr};
    p.hi=read(QH,i); p.hj=read(RH,j);
    p.h=args.support==11 ? p.hi : args.support==12 ? p.hj : args.support==13 ?
        (p.hi+p.hj)/S(2) : ScalarValue<S>::get(p.hi)>=ScalarValue<S>::get(p.hj) ? p.hi : p.hj;
    kernel<T,D>(p.dx,p.h,args.kernel,p.raw_w,p.raw_g,p.lap);
    if (args.support==14 || args.support==15) {
        S wi,wj,gi[D],gj[D],li,lj;
        kernel<T,D>(p.dx,p.hi,args.kernel,wi,gi,li);
        kernel<T,D>(p.dx,p.hj,args.kernel,wj,gj,lj);
        p.raw_w=(wi+wj)/S(2);
        for (int d=0; d<D; ++d) p.raw_g[d]=(gi[d]+gj[d])/S(2);
        if (args.support==15) p.lap=(li+lj)/S(2);
    }
    p.crk=arrays.values[A]; p.renormalized=arrays.values[R];
    p.coefficient_A=read(A,i,T(1)); p.polynomial=S(1);
    for (int d=0; d<D; ++d) {
        p.coefficient_B[d]=read(B,i*D+d);
        p.grad_A[d]=read(GA,i*D+d);
        p.polynomial+=p.coefficient_B[d]*p.dx[d];
        for (int c=0; c<D; ++c) {
            p.grad_B[d*D+c]=read(GB,i*D*D+d*D+c);
            p.matrix[d*D+c]=read(R,i*D*D+d*D+c,d==c ? T(1):T(0));
        }
    }
    p.w=p.coefficient_A*p.polynomial*p.raw_w;
    for (int d=0; d<D; ++d) {
        S product=S(0);
        for (int c=0; c<D; ++c) product+=p.grad_B[d*D+c]*p.dx[c];
        p.before_R[d]=p.crk ? p.coefficient_A*p.coefficient_B[d]*p.raw_w+
            p.coefficient_A*p.polynomial*p.raw_g[d]+p.polynomial*p.raw_w*p.grad_A[d]+
            p.coefficient_A*p.raw_w*product : p.raw_g[d];
    }
    for (int d=0; d<D; ++d) {
        p.gradient[d]=S(0);
        for (int c=0; c<D; ++c) p.gradient[d]+=p.matrix[d*D+c]*p.before_R[c];
    }
    p.mass=read(MASS,j,T(1)); p.rhoi=read(QDENSITY,i,T(1)); p.rhoj=read(DENSITY,j,T(1));
    p.volume=arrays.values[VOLUME] ? read(VOLUME,j) : p.mass/p.rhoj;
    p.query_omega=read(QO,i,T(1)); p.reference_omega=read(RO,j,T(1));
    for (int c=0; c<args.fields; ++c) {
        p.fi[c]=read(QVALUE,i*args.fields+c);
        p.fj[c]=read(RVALUE,j*args.fields+c);
        if (args.operation!=1) p.fj[c]=p.fj[c]/p.reference_omega;
        if (args.operation>=3 && args.operation<=5) p.fi[c]=p.fi[c]/p.query_omega;
        if (args.operation==1) p.weighted[c]=p.volume*p.fj[c];
        else if (args.operation==5) p.weighted[c]=(p.fj[c]-p.fi[c])*(args.scheme==2 ?
            p.mass*p.rhoi/(p.rhoj*p.rhoj) : p.volume);
        else if (args.scheme==2) p.weighted[c]=p.mass*(p.fi[c]/p.rhoi+p.rhoi*p.fj[c]/(p.rhoj*p.rhoj));
        else p.weighted[c]=p.volume*(args.scheme==1 ? p.fj[c] : args.scheme==3 ? p.fj[c]-p.fi[c] : p.fj[c]+p.fi[c]);
        if (consistent && args.operation==3 && args.scheme!=2) p.weighted[c]=p.weighted[c]*p.rhoj/p.rhoi;
    }
    return p;
}

// Directed pair outputs before row finalization. Moment tensors use the same
// derivative-first axes as the existing CRK preparation policy.
template <class T, int D, class S>
SPH_HD void forward(const RowArgs<T,D>& a, const Pair<S,D>& p, S* out) {
    if (a.operation==Volume) { out[0]+=p.raw_w; return; }
    if (a.operation==Consistency) {
        out[0]+=p.mass*p.volume*p.w;
        out[1]+=p.volume*p.volume*p.w; return;
    }
    if (a.operation==Moments) {
        constexpr int m1=1,m2=m1+D,dm0=m2+D*D,dm1=dm0+D,dm2=dm1+D*D;
        out[0]+=p.volume*p.raw_w;
        for (int d=0; d<D; ++d) {
            out[m1+d]+=p.volume*p.raw_w*p.dx[d]; out[dm0+d]+=p.volume*p.raw_g[d];
            for (int c=0; c<D; ++c) {
                out[m2+d*D+c]+=p.volume*p.raw_w*p.dx[d]*p.dx[c];
                out[dm1+d*D+c]+=p.volume*(p.dx[d]*p.raw_g[c]+S(d==c)*p.raw_w);
                for (int e=0; e<D; ++e) out[dm2+e*D*D+d*D+c]+=p.volume*(
                    p.raw_g[e]*p.dx[d]*p.dx[c]+p.raw_w*(S(d==e)*p.dx[c]+S(c==e)*p.dx[d]));
            }
        }
        return;
    }
    if (a.operation==7) {
        for (int d=0; d<D; ++d) for (int c=0; c<D; ++c) out[d*D+c]-=p.volume*p.dx[d]*p.gradient[c];
    } else if (a.operation==1) {
        for (int c=0; c<a.fields; ++c) out[c]+=p.weighted[c]*p.w;
    } else if (a.operation==2) {
        for (int c=0; c<a.fields; ++c) for (int d=0; d<D; ++d) out[c*D+d]+=p.weighted[c]*p.gradient[d];
    } else if (a.operation==3) {
        const int components=a.fields/D;
        for (int c=0; c<components; ++c) for (int d=0; d<D; ++d)
            out[c]+=p.gradient[d]*p.weighted[a.dot_mode ? d*components+c : c*D+d];
    } else if (a.operation==4) {
        if constexpr (D==2) {
            const int stride=a.fields/2;
            for (int c=0; c<stride; ++c) out[c]+=p.gradient[0]*p.weighted[stride+c]-p.gradient[1]*p.weighted[c];
        } else if constexpr (D==3) {
            const int stride=a.fields/3;
            for (int c=0; c<stride; ++c) for (int d=0; d<3; ++d)
                out[d*stride+c]+=p.gradient[(d+1)%3]*p.weighted[((d+2)%3)*stride+c]-p.gradient[(d+2)%3]*p.weighted[((d+1)%3)*stride+c];
        }
    } else if (a.operation==5) {
        if (a.positive_divergence) {
            S gate=S(0);
            for (int d=0;d<D;++d) gate+=p.dx[d]*p.weighted[d];
            if (ScalarValue<S>::get(gate)<0) return;
        }
        const int fields=a.positive_divergence ? D:a.fields;
        if (a.laplacian_mode==1) {
            for (int c=0; c<fields; ++c) out[c]+=p.weighted[c]*p.lap;
        } else if (ScalarValue<S>::get(p.distance)>0) {
            const S denominator=p.distance+S(1e-8)*p.h;
            S n[D],projection=S(0);
            for (int d=0; d<D; ++d) { n[d]=p.dx[d]/denominator; projection+=p.gradient[d]*n[d]; }
            if (a.laplacian_mode==3) {
                for (int base=0; base<fields; base+=D) {
                    S z=S(0); for (int d=0; d<D; ++d) z+=p.weighted[base+d]*n[d];
                    for (int d=0; d<D; ++d) out[base+d]+=(p.weighted[base+d]-S(D+2)*z*n[d])*projection/denominator;
                }
            } else {
                const S divisor=p.distance+S(a.laplacian_mode==4 ? 1e-12 : 1e-8)*p.h;
                for (int c=0; c<fields; ++c) out[c]-=S(2)*p.weighted[c]*projection/divisor;
            }
        }
    }
}
} // namespace sph_native::differential
