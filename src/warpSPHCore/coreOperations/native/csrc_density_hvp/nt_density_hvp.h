#pragma once
#include "execution/dispatch.h"
#include "execution/csrView.h"
#include "execution/neighborhoodAdjoint.h"
#include "math/queryGeometry.h"
#include "kernels/radial.h"
#include "math/pow.h"
#include "util/directionality.h"

namespace sph_native::density_hvp {
using neighborhood::Owner;

// Preserve the public Warp helper's regularized Hessian convention, including
// its origin branch. This is an explicit position block, not double backward.
template <class T,int D>
SPH_HD void hessian(const T (&dx)[D],T r,T h,int kernel,T (&out)[D*D]) {
    for (int c=0;c<D*D;++c) out[c]=0;
    if (h<=0 || r>h) return;
    const T q=r/h,eps=std::is_same_v<T,float> ? T(1e-7):T(1e-15);
    const auto radial=radial_shape<T,D>(q,kernel);
    const T scale=normalization<T,D>(kernel)/power<T,D>(h,D);
    const T first=scale*radial.d1/h,second=scale*radial.d2/(h*h);
    const T square=r*r+eps*eps*h*h,cube=r*r*r+eps*eps*eps*h*h*h;
    for (int d=0;d<D;++d) for (int c=0;c<D;++c) {
        const T outer=dx[d]*dx[c];
        const T a=q<eps && d==c ? T(1):outer/square;
        const T b=-outer/cube+(d==c ? T(1)/(r+eps*eps*h):T(0));
        out[d*D+c]=a*second+b*first;
    }
}

template <class T,int D> struct Pair {
    RowArgs<T,D> args;
    int64_t row_start;
    SPH_HD bool matrix(int64_t i,int64_t j,T (&H)[D*D]) const {
        if (i+row_start==j || !query_allowed(args.qkind[i],args.direction) ||
            !reference_allowed(args.rkind[j],args.direction)) return false;
        T dx[D];
        const T r=QueryGeometry<T,D>(args.qx+i*D,args.lo,args.hi,args.periodic).distance(args.rx+j*D,dx);
        hessian<T,D>(dx,r,args.radius(i,j),args.kernel,H);
        if (args.support==15) {
            T other[D*D];
            hessian<T,D>(dx,r,args.qh[i],args.kernel,H);
            hessian<T,D>(dx,r,args.rh[j],args.kernel,other);
            for (int c=0;c<D*D;++c) H[c]=(H[c]+other[c])/T(2);
        }
        return true;
    }
    SPH_HD void operator()(int64_t i) const {
        T sum[D]={};
        for (int k=0;k<args.counts[i];++k) {
            const auto j=args.neighbors[args.offsets[i]+k];
            T H[D*D]; if (!matrix(i,j,H)) continue;
            for (int d=0;d<D;++d) for (int c=0;c<D;++c)
                sum[d]+=args.mass[j]*H[d*D+c]*(args.qvalue[i*D+c]-(args.rvalue ? args.rvalue[j*D+c]:T(0)));
        }
        for (int d=0;d<D;++d) args.output[i*D+d]=sum[d];
    }
};

template <class T,int D> struct Pullback {
    Pair<T,D> pair;
    const T* cotangent;
    T *query_bar,*reference_bar,*mass_bar;
    template <Owner Side> SPH_HD int64_t prepare(int64_t owner) const {return owner;}
    template <Owner Side> SPH_HD auto identity() const {return neighborhood::Sums<T,D+1>{};}
    template <Owner Side,class State>
    SPH_HD void interact(int64_t owner,int64_t,neighborhood::Neighbor neighbor,State& sum) const {
        const int64_t i=Side==Owner::Query ? owner:neighbor.reference;
        const int64_t j=Side==Owner::Query ? neighbor.reference:owner;
        const auto& a=pair.args;
        T H[D*D]; if (!pair.matrix(i,j,H)) return;
        for (int c=0;c<D;++c) {
            T v=0; for (int d=0;d<D;++d) v+=cotangent[i*D+d]*H[d*D+c];
            sum.values[c]+=(Side==Owner::Query ? T(1):T(-1))*a.mass[j]*v;
            if constexpr (Side==Owner::Reference)
                sum.values[D]+=v*(a.qvalue[i*D+c]-(a.rvalue ? a.rvalue[j*D+c]:T(0)));
        }
    }
    template <Owner Side,class State> SPH_HD void finish(int64_t owner,const State& sum) const {
        auto* output=Side==Owner::Query ? query_bar:reference_bar;
        if (output) for (int d=0;d<D;++d) output[owner*D+d]=sum.values[d];
        if constexpr (Side==Owner::Reference) if (mass_bar) mass_bar[owner]=sum.values[D];
    }
};
struct Buffers {
    Tensor cotangent;
    Tensors transpose,gradients;
    int64_t row_start;
};
struct Rows {
    const Buffers& buffers;
    template <class T,int D,class Launch>
    void operator()(int64_t n,RowArgs<T,D> a,const Launch& launch) const {
        const Pair<T,D> pair{a,buffers.row_start};
        if (!buffers.cotangent.defined()) {launch(n,pair);return;}
        const Pullback<T,D> pullback{pair,ptr<T>(buffers.cotangent),
            buffers.gradients[0].mutable_data_ptr<T>(),buffers.gradients[1].mutable_data_ptr<T>(),
            buffers.gradients[2].mutable_data_ptr<T>()};
        const neighborhood::CsrView csr{a.offsets,a.counts,a.neighbors};
        launch(n,neighborhood::OwnedAdjointRow<Owner::Query,decltype(pullback),decltype(csr)>{pullback,csr});
        const auto& t=buffers.transpose;
        const neighborhood::CsrView csc{index_view(t[0]),ptr<int32_t>(t[1]),index_view(t[2])};
        launch(a.references,neighborhood::OwnedAdjointRow<Owner::Reference,decltype(pullback),decltype(csc)>{pullback,csc});
    }
};
} // namespace sph_native::density_hvp
