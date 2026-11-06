#pragma once
#include "coreOperations/native/csrc_differential/nt_differential_pair.h"
#include "execution/csrView.h"
#include "kernels/explicit_value_derivatives.h"
#include "execution/neighborhoodAdjoint.h"

namespace sph_native::crk_volume_jvp {
using neighborhood::Owner;
// Reverse the explicit Gather product. The gradient uses Warp's custom
// Hessian; the support derivative is the separately taped grad-H expression.
using warp_compat::support_value;
using warp_compat::support_pullback;
struct Buffers {OptionalTensors directions;Tensors transpose,gradients;Tensor cotangent,volumes;};
template<class T,int D> struct Pullback {
    RowArgs<T,D> args;
    const T *tq,*tr,*th,*volumes,*cotangent;
    T *bars[9];
    template<Owner Side> SPH_HD int64_t prepare(int64_t owner) const{return owner;}
    template<Owner Side> SPH_HD auto identity() const{return neighborhood::Sums<T,2*D+3>{};}
    template<Owner Side,class State> SPH_HD void interact(int64_t owner,int64_t,neighborhood::Neighbor neighbor,State& sum) const {
        const auto& a=args;const int64_t i=Side==Owner::Query?owner:neighbor.reference,j=Side==Owner::Query?neighbor.reference:owner;
        if(!query_allowed(a.qkind[i],a.direction)||!reference_allowed(a.rkind[j],a.direction))return;
        T dx[D],dt[D];const T r=QueryGeometry<T,D>(a.qx+i*D,a.lo,a.hi,a.periodic).distance(a.rx+j*D,dx),h=a.qh[i];
        for(int c=0;c<D;++c)dt[c]=(tq?tq[i*D+c]:T(0))-(tr?tr[j*D+c]:T(0));
        const T dh=th?th[i]:T(0),v=volumes[i],incoming=-v*(v*cotangent[i]);
        T w=0,g[D]={},wx[D]={},wh=0,H[D*D]={},gh[D]={};
        warp_compat::force_derivatives<T,D>(dx,r,h,a.kernel,w,g,wx,wh,H,gh);
        wh=support_value<T,D>(r,h,a.kernel);
        T bx[D]={},bh=0,dw=0;
        for(int c=0;c<D;++c)dw+=g[c]*dt[c];
        dw+=wh*dh;
        support_pullback<T,D>(dx,r,h,a.kernel,incoming*dh,bx,bh);
        for(int c=0;c<D;++c) {
            const T bg=incoming*dt[c];bh+=bg*gh[c];
            for(int e=0;e<D;++e)bx[e]+=H[c*D+e]*bg;
        }
        for(int c=0;c<D;++c) {
            sum.values[c]+=Side==Owner::Query?bx[c]:-bx[c];
            sum.values[D+c]+=Side==Owner::Query?incoming*g[c]:-incoming*g[c];
        }
        if constexpr(Side==Owner::Query){sum.values[2*D]+=bh;sum.values[2*D+1]+=incoming*wh;sum.values[2*D+2]+=dw;}
    }
    template<Owner Side,class State> SPH_HD void finish(int64_t owner,const State& sum) const {
        const int pos=Side==Owner::Query?0:1,tangent=Side==Owner::Query?4:5;
        for(int c=0;c<D;++c){bars[pos][owner*D+c]=sum.values[c];bars[tangent][owner*D+c]=sum.values[D+c];}
        if constexpr(Side==Owner::Query){
            bars[2][owner]=sum.values[2*D];bars[6][owner]=sum.values[2*D+1];
            const T ds=sum.values[2*D+2],v=volumes[owner],cot=cotangent[owner];
            bars[8][owner]=query_allowed(args.qkind[owner],args.direction)?(-ds*v)*cot+(-ds)*(v*cot):T(0);
        }
    }
};
template<class T,int D> struct ProductRow {
    RowArgs<T,D> a;
    const T *tq,*tr,*th,*volumes;
    SPH_HD void operator()(int64_t i) const {
        T sum=0;
        if(!query_allowed(a.qkind[i],a.direction)){a.output[i]=T(0);return;}
        for(int64_t k=0;k<a.counts[i];++k){
            const auto j=a.neighbors[a.offsets[i]+k];if(!reference_allowed(a.rkind[j],a.direction))continue;
            T dx[D];const T r=QueryGeometry<T,D>(a.qx+i*D,a.lo,a.hi,a.periodic).distance(a.rx+j*D,dx),h=a.qh[i];
            T w=0,g[D]={},wx[D]={},wh=0,H[D*D]={},gh[D]={};
            warp_compat::force_derivatives<T,D>(dx,r,h,a.kernel,w,g,wx,wh,H,gh);
            T dot=0;for(int c=0;c<D;++c)dot+=g[c]*((tq?tq[i*D+c]:T(0))-(tr?tr[j*D+c]:T(0)));
            sum+=dot+support_value<T,D>(r,h,a.kernel)*(th?th[i]:T(0));
        }
        a.output[i]=-sum*volumes[i]*volumes[i];
    }
};
struct Rows {
    const Buffers& b;
    template<class T,int D,class Launch> void operator()(int64_t n,RowArgs<T,D> a,const Launch& launch) const {
        if(!b.cotangent.defined()) {
            launch(n,ProductRow<T,D>{a,b.directions[0]?ptr<T>(*b.directions[0]):nullptr,
                b.directions[1]?ptr<T>(*b.directions[1]):nullptr,b.directions[2]?ptr<T>(*b.directions[2]):nullptr,ptr<T>(b.volumes)});
            return;
        }
        Pullback<T,D> p{a,b.directions[0]?ptr<T>(*b.directions[0]):nullptr,
            b.directions[1]?ptr<T>(*b.directions[1]):nullptr,b.directions[2]?ptr<T>(*b.directions[2]):nullptr,
            ptr<T>(b.volumes),ptr<T>(b.cotangent),{}};
        for(int k=0;k<9;++k)p.bars[k]=b.gradients[k].mutable_data_ptr<T>();
        const neighborhood::CsrView csr{a.offsets,a.counts,a.neighbors};
        launch(n,neighborhood::OwnedAdjointRow<Owner::Query,decltype(p),decltype(csr)>{p,csr});
        const auto& t=b.transpose;const neighborhood::CsrView csc{index_view(t[0]),ptr<int32_t>(t[1]),index_view(t[2])};
        launch(a.references,neighborhood::OwnedAdjointRow<Owner::Reference,decltype(p),decltype(csc)>{p,csc});
    }
};
} // namespace sph_native::crk_volume_jvp
