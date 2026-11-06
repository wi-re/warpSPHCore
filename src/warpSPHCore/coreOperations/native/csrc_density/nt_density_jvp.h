#pragma once
#include "execution/dispatch.h"
#include "execution/csrView.h"
#include "execution/neighborhoodAdjoint.h"
#include "math/queryGeometry.h"
#include "kernels/value.h"
#include "kernels/explicit_value_derivatives.h"
#include "util/directionality.h"

namespace sph_native::density_jvp {
using neighborhood::Owner;
template<class T,int D> struct Kernel {
    T x[D],r,h,w,g[D],wh,H[D*D],gh[D];
    SPH_HD Kernel(const T (&dx)[D],T radius,T support,int name):r(radius),h(support),w(0),g{},wh(0),H{},gh{} {
        for(int c=0;c<D;++c)x[c]=dx[c];
        T wx[D]={};warp_compat::force_derivatives<T,D>(x,r,h,name,w,g,wx,wh,H,gh);
        w=kernel_value<T,D>(r,h,name);wh=warp_compat::support_value<T,D>(r,h,name);
    }
    SPH_HD T tangent(const T (&dt)[D],T dh) const {
        T dot=0;for(int c=0;c<D;++c)dot+=g[c]*dt[c];return dot+wh*dh;
    }
    SPH_HD void backward(int name,const T (&dt)[D],T dh,T bw,T bd,T (&bx)[D],T& bh,T (&btx)[D],T& bth) const {
        warp_compat::support_pullback<T,D>(x,r,h,name,dh*bd,bx,bh);
        for(int c=0;c<D;++c){
            const T bg=dt[c]*bd;bh+=warp_compat::rounded_scalar_term(gh[c]*bg);
            for(int e=0;e<D;++e)bx[e]+=H[c*D+e]*bg;
            btx[c]+=g[c]*bd;
        }
        bth+=wh*bd;
        warp_compat::value_pullback<T,D>(x,r,h,name,bw,bx,bh);
    }
};
struct Buffers {OptionalTensors directions;Tensors transpose,outputs;Tensor cotangent;};
template<class T,int D> struct Pair {
    RowArgs<T,D> a;const T *tq,*tr,*thq,*thr,*tm,*cot;
    T* bars[10];
    SPH_HD T direction(const T* p,int64_t at) const{return p?p[at]:T(0);}
    template<class State> SPH_HD T interact(int64_t i,int64_t j,State& sum,bool query,bool reverse) const {
        if(!query_allowed(a.qkind[i],a.direction)||!reference_allowed(a.rkind[j],a.direction))return T(0);
        T dx[D],dt[D];const T r=QueryGeometry<T,D>(a.qx+i*D,a.lo,a.hi,a.periodic).distance(a.rx+j*D,dx);
        for(int c=0;c<D;++c)dt[c]=direction(tq,i*D+c)-direction(tr,j*D+c);
        const T hi=a.qh[i],hj=a.rh[j],ti=direction(thq,i),tj=direction(thr,j),mass=a.mass[j],dm=direction(tm,j);
        const bool average=a.support==14||a.support==15;
        const T h=a.support==11?hi:a.support==12?hj:a.support==13?(hi+hj)/T(2):hi>hj?hi:hj;
        const T dh=a.support==11?ti:a.support==12?tj:a.support==13?(ti+tj)/T(2):hi>=hj?ti:tj;
        T w=0,dw=0,bx[D]={},btx[D]={},bhi=0,bhj=0,bti=0,btj=0;
        if(average){
            const Kernel<T,D> left(dx,r,hi,a.kernel),right(dx,r,hj,a.kernel);
            w=(left.w+right.w)/T(2);
            T dl=0,dr=0;for(int c=0;c<D;++c){dl+=left.g[c]*dt[c];dr+=right.g[c]*dt[c];}
            dw=((dl+dr)+left.wh*ti+right.wh*tj)*T(.5);
            if(reverse){
                const T bw=dm*cot[i]/T(2),bd=mass*cot[i]*T(.5);
                right.backward(a.kernel,dt,tj,bw,bd,bx,bhj,btx,btj);
                left.backward(a.kernel,dt,ti,bw,bd,bx,bhi,btx,bti);
            }
        }else{
            const Kernel<T,D> kernel(dx,r,h,a.kernel);w=kernel.w;dw=kernel.tangent(dt,dh);
            if(reverse){
                T bh=0,bth=0;kernel.backward(a.kernel,dt,dh,dm*cot[i],mass*cot[i],bx,bh,btx,bth);
                if(a.support==11){bhi=bh;bti=bth;}
                else if(a.support==12){bhj=bh;btj=bth;}
                else if(a.support==13){bhi=bhj=bh/T(2);bti=btj=bth/T(2);}
                else{if(hi>hj)bhi=bh;else bhj=bh;if(hi>=hj)bti=bth;else btj=bth;}
            }
        }
        if(reverse){
            for(int c=0;c<D;++c){sum.values[c]+=query?bx[c]:-bx[c];sum.values[D+c]+=query?btx[c]:-btx[c];}
            sum.values[2*D]+=query?bhi:bhj;sum.values[2*D+1]+=query?bti:btj;
            if(!query){sum.values[2*D+2]+=dw*cot[i];sum.values[2*D+3]+=w*cot[i];}
        }
        return dm*w+mass*dw;
    }
};
template<class T,int D> struct Pullback {
    Pair<T,D> pair;
    template<Owner Side>SPH_HD int64_t prepare(int64_t i) const{return i;}
    template<Owner Side>SPH_HD auto identity() const{return neighborhood::Sums<T,2*D+4>{};}
    template<Owner Side,class State>SPH_HD void interact(int64_t owner,int64_t,neighborhood::Neighbor neighbor,State& sum) const {
        pair.interact(Side==Owner::Query?owner:neighbor.reference,Side==Owner::Query?neighbor.reference:owner,sum,Side==Owner::Query,true);
    }
    template<Owner Side,class State>SPH_HD void finish(int64_t owner,const State& sum) const {
        const bool q=Side==Owner::Query;const auto& bars=pair.bars;
        for(int c=0;c<D;++c){bars[q?0:1][owner*D+c]=sum.values[c];bars[q?5:6][owner*D+c]=sum.values[D+c];}
        bars[q?2:3][owner]=sum.values[2*D];bars[q?7:8][owner]=sum.values[2*D+1];
        if constexpr(Side==Owner::Reference){bars[4][owner]=sum.values[2*D+2];bars[9][owner]=sum.values[2*D+3];}
    }
};
template<class T,int D>struct ProductRow {
    Pair<T,D> pair;
    SPH_HD void operator()(int64_t i) const {
        T total=0;neighborhood::Sums<T,2*D+4> unused{};
        for(int64_t k=0;k<pair.a.counts[i];++k)total+=pair.interact(i,pair.a.neighbors[pair.a.offsets[i]+k],unused,true,false);
        pair.a.output[i]=total;
    }
};
struct Rows {
    const Buffers& b;
    template<class T,int D,class Launch>void operator()(int64_t n,RowArgs<T,D> a,const Launch& launch) const {
        auto ptr_or_null=[&](int k){return b.directions[k]?ptr<T>(*b.directions[k]):nullptr;};
        Pair<T,D> pair{a,ptr_or_null(0),ptr_or_null(1),ptr_or_null(2),ptr_or_null(3),ptr_or_null(4),b.cotangent.defined()?ptr<T>(b.cotangent):nullptr,{}};
        if(!b.cotangent.defined()){launch(n,ProductRow<T,D>{pair});return;}
        for(int k=0;k<10;++k)pair.bars[k]=b.outputs[k].mutable_data_ptr<T>();
        const Pullback<T,D> pull{pair};const neighborhood::CsrView csr{a.offsets,a.counts,a.neighbors};
        launch(n,neighborhood::OwnedAdjointRow<Owner::Query,decltype(pull),decltype(csr)>{pull,csr});
        const auto& t=b.transpose;const neighborhood::CsrView csc{index_view(t[0]),ptr<int32_t>(t[1]),index_view(t[2])};
        launch(a.references,neighborhood::OwnedAdjointRow<Owner::Reference,decltype(pull),decltype(csc)>{pull,csc});
    }
};
}
