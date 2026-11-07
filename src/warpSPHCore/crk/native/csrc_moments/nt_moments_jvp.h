#pragma once
#include "execution/explicitForceProduct.h"

namespace sph_native::crk_moments_jvp {
using neighborhood::Owner;
struct Buffers { OptionalTensors directions; Tensors transpose, outputs; Tensor cotangent, volumes; };
template<int D> constexpr int width = 1 + 2*D + 2*D*D + D*D*D;
template<class T> union Scratch {
    explicit_ad::Trace<T,1536> pair;
    explicit_ad::Trace<T,512> force;
    SPH_HD Scratch() {}
};

template<class T,int D> struct Pair {
    force_jvp::Pair<T,D,7> kernel;
    const T *volumes, *volume_direction, *cotangent;
    T* output[10];

    // The value and gradient primitives have distinct reference adjoints.
    // Reuse the force-product kernel chain, then add the value-product chain.
    SPH_HD void value_backward(const T(&x)[D],T r,T hi,T hj,const T(&dt)[D],T ti,T tj,
            T bw,T bdw,force_jvp::Bars<T,D>& bars) const {
        T bx[D]={},bt[D]={},bhi=0,bhj=0,bti=0,btj=0;
        auto one=[&](T h,T dh,T scale,T& bh,T& bth) {
            if(kernel.geometry()) {
                const warp_compat::ValueProductKernel<T,D> k(x,r,h,kernel.a.kernel);
                k.backward(kernel.a.kernel,dt,dh,bw*scale,bdw*scale,bx,bh,bt,bth);
            } else warp_compat::value_pullback<T,D>(x,r,h,kernel.a.kernel,bw*scale,bx,bh);
        };
        if(kernel.a.support==14||kernel.a.support==15) {
            one(hj,tj,T(.5),bhj,btj);one(hi,ti,T(.5),bhi,bti);
        } else {
            const int mode=kernel.a.support;
            const T h=mode==11?hi:mode==12?hj:mode==13?(hi+hj)/T(2):hi>hj?hi:hj;
            const T dh=mode==11?ti:mode==12?tj:mode==13?(ti+tj)/T(2):hi>=hj?ti:tj;
            T bh=0,bth=0;one(h,dh,T(1),bh,bth);
            if(mode==11){bhi=bh;bti=bth;}else if(mode==12){bhj=bh;btj=bth;}
            else if(mode==13){bhi=bhj=bh/T(2);bti=btj=bth/T(2);}
            else{if(hi>hj)bhi=bh;else bhj=bh;if(hi>=hj)bti=bth;else btj=bth;}
        }
        for(int c=0;c<D;++c){bars.add(0,c,1,bx[c]);bars.add(1,c,1,-bx[c]);bars.add(26,c,1,bt[c]);bars.add(27,c,1,-bt[c]);}
        bars.add(2,0,1,bhi);bars.add(3,0,1,bhj);bars.add(28,0,1,bti);bars.add(29,0,1,btj);
    }

    template<bool Reverse> SPH_HD force_jvp::Bars<T,D> interact(int64_t i,int64_t j,T* sum) const {
        force_jvp::Bars<T,D> bars{};const auto& a=kernel.a;
        if(!query_allowed(a.qkind[i],a.direction)||!reference_allowed(a.rkind[j],a.direction))return bars;
        const bool geometry=kernel.geometry();
        if(!geometry&&!volume_direction)return bars;
        T x[D],dt[D];const T r=QueryGeometry<T,D>(a.qx+i*D,a.lo,a.hi,a.periodic).distance(a.rx+j*D,x);
        for(int c=0;c<D;++c)dt[c]=kernel.relative_tangent(i,j,c);
        const T hi=a.qh[i],hj=a.rh[j],ti=kernel.tangent(2,i),tj=kernel.tangent(3,j);
        T w=0,dw=0,g[D]={},dg[D]={};kernel.kernel(x,r,hi,hj,dt,ti,tj,w,dw,g,dg);
        using S=pair_ad::Number<T,Reverse,1536>;
        using Storage=std::conditional_t<Reverse,Scratch<T>,char>;Storage storage;
        explicit_ad::Trace<T,1536>* trace=nullptr;
        if constexpr(Reverse)trace=new(&storage.pair)explicit_ad::Trace<T,1536>;
        auto leaf=[&](T v,int slot,int c,bool active=true){return S::leaf(v,0,trace,slot,c,Reverse&&active);};
        S xx[D],dd[D],G[D],dG[D];
        for(int c=0;c<D;++c){
            xx[c]=leaf(a.qx[i*D+c],0,c)-leaf(a.rx[j*D+c],1,c);xx[c].v=x[c];
            dd[c]=leaf(kernel.tangent(0,i*D+c),26,c,kernel.t[0])-leaf(kernel.tangent(1,j*D+c),27,c,kernel.t[1]);
            G[c]=leaf(g[c],54,c);dG[c]=leaf(dg[c],55,c,kernel.geometry());
        }
        const S V=leaf(volumes[j],18,0),dV=leaf(volume_direction?volume_direction[j]:T(0),44,0,volume_direction);
        const S W=leaf(w,52,0),dW=leaf(dw,53,0,kernel.geometry());
        // A fixed-geometry volume product is the ordinary linear moment map
        // applied to dV. Do not tape unused 0*Inf geometry terms at cutoffs.
        const S VW=geometry?V*W:S(0),dVW=geometry?dV*W+V*dW:dV*W;
        auto emit=[&](int c,S value){if constexpr(Reverse)value.seed(cotangent[i*width<D>+c]);else sum[c]+=value.v;};
        int at=0;emit(at++,dVW);
        for(int c=0;c<D;++c)emit(at++,geometry?dd[c]*VW+xx[c]*dVW:xx[c]*dVW);
        for(int c=0;c<D;++c)for(int e=0;e<D;++e)emit(at++,geometry?(dd[c]*xx[e]+xx[c]*dd[e])*VW+(xx[c]*xx[e])*dVW:(xx[c]*xx[e])*dVW);
        for(int c=0;c<D;++c)emit(at++,geometry?dV*G[c]+V*dG[c]:dV*G[c]);
        for(int c=0;c<D;++c)for(int e=0;e<D;++e){const S eye(c==e?T(1):T(0));
            emit(at++,geometry?dV*(xx[c]*G[e]+W*eye)+V*(dd[c]*G[e]+xx[c]*dG[e]+dW*eye):dV*(xx[c]*G[e]+W*eye));}
        // Keep the reference alpha/beta/gamma construction order and its
        // gamma-leading flattened output layout.
        for(int alpha=0;alpha<D;++alpha)for(int beta=0;beta<D;++beta)for(int gamma=0;gamma<D;++gamma){
            const S da(beta==gamma?T(1):T(0)),db(alpha==gamma?T(1):T(0));
            const S grad=xx[alpha]*xx[beta]*G[gamma];
            const S dgrad=geometry?dd[alpha]*xx[beta]*G[gamma]+xx[alpha]*dd[beta]*G[gamma]+xx[alpha]*xx[beta]*dG[gamma]:S(0);
            const S delta_a=xx[alpha]*da,delta_b=db*xx[beta],ddelta_a=geometry?dd[alpha]*da:S(0),ddelta_b=geometry?db*dd[beta]:S(0);
            const S term=W*(delta_a+delta_b),dterm=geometry?dW*(delta_a+delta_b)+W*(ddelta_a+ddelta_b):S(0);
            emit(at+gamma*D*D+alpha*D+beta,geometry?dV*(grad+term)+V*(dgrad+dterm):dV*(grad+term));
        }
        if constexpr(Reverse){
            trace->reverse();T bw=0,bdw=0,bg[D]={},bdg[D]={};
            for(int k=0;k<trace->inputs;++k){const auto l=trace->leaves[k];const T bar=trace->nodes[l.node].bar;
                if(l.slot==52)bw+=bar;else if(l.slot==53)bdw+=bar;else if(l.slot==54)bg[l.at]+=bar;else if(l.slot==55)bdg[l.at]+=bar;else bars.add(l.slot,l.at,1,bar);}
            auto* force=new(&storage.force)explicit_ad::Trace<T,512>;
            kernel.template reverse_kernel<true>(x,r,hi,hj,dt,ti,tj,T(0),T(0),bg,bdg,bars,force);
            value_backward(x,r,hi,hj,dt,ti,tj,bw,bdw,bars);
        }
        return bars;
    }
};

template<class T,int D> struct Pullback {
    Pair<T,D> pair;
    template<Owner Side> SPH_HD int64_t prepare(int64_t i)const{return i;}
    template<Owner Side> SPH_HD auto identity()const{return neighborhood::Sums<T,2*(Side==Owner::Query?force_jvp::Layout<D>::Q:force_jvp::Layout<D>::R)>{};}
    template<Owner Side,class State> SPH_HD void interact(int64_t owner,int64_t,neighborhood::Neighbor n,State& sum)const{
        const auto bars=pair.template interact<true>(Side==Owner::Query?owner:n.reference,Side==Owner::Query?n.reference:owner,nullptr);
        for(int c=0;c<2*(Side==Owner::Query?force_jvp::Layout<D>::Q:force_jvp::Layout<D>::R);++c)sum.values[c]+=(Side==Owner::Query?bars.q:bars.r)[c];
    }
    template<Owner Side,class State> SPH_HD void finish(int64_t owner,const State& sum)const{
        constexpr int slots[10]={0,1,2,3,18,26,27,28,29,44};
        for(int k=0;k<10;++k)if(force_jvp::Layout<D>::query(slots[k])==(Side==Owner::Query)){
            const int count=force_jvp::Layout<D>::width(slots[k],1),at=force_jvp::Layout<D>::offset(slots[k],1);
            for(int c=0;c<count;++c)pair.output[k][owner*count+c]=sum.values[at+c];
        }
    }
};
template<class T,int D> struct ProductRow {
    Pair<T,D> pair;
    SPH_HD void operator()(int64_t i)const{
        T sum[width<D>]={};const auto& a=pair.kernel.a;
        for(int64_t k=0;k<a.counts[i];++k)pair.template interact<false>(i,a.neighbors[a.offsets[i]+k],sum);
        for(int c=0;c<width<D>;++c)a.output[i*width<D>+c]=sum[c];
    }
};
struct Rows {
    const Buffers& b;
    template<class T,int D,class Launch>void operator()(int64_t n,RowArgs<T,D>a,const Launch& launch)const{
        a.fields=1;force_jvp::Pair<T,D,7> helper{a,{}, {},nullptr,{}};
        for(int k=0;k<4;++k)helper.t[k]=b.directions[k]?ptr<T>(*b.directions[k]):nullptr;
        Pair<T,D> pair{helper,ptr<T>(b.volumes),b.directions[4]?ptr<T>(*b.directions[4]):nullptr,b.cotangent.defined()?ptr<T>(b.cotangent):nullptr,{}};
        if(!b.cotangent.defined()){launch(n,ProductRow<T,D>{pair});return;}
        for(int k=0;k<10;++k)pair.output[k]=b.outputs[k].mutable_data_ptr<T>();
        const Pullback<T,D> pull{pair};const neighborhood::CsrView csr{a.offsets,a.counts,a.neighbors};
        launch(n,neighborhood::OwnedAdjointRow<Owner::Query,decltype(pull),decltype(csr)>{pull,csr});
        const auto& t=b.transpose;const neighborhood::CsrView csc{index_view(t[0]),ptr<int32_t>(t[1]),index_view(t[2])};
        launch(a.references,neighborhood::OwnedAdjointRow<Owner::Reference,decltype(pull),decltype(csc)>{pull,csc});
    }
};
} // namespace sph_native::crk_moments_jvp
