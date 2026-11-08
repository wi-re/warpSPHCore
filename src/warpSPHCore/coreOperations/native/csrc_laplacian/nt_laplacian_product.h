#pragma once
#include "execution/explicitForceProduct.h"
#include "kernels/explicit_laplacian_derivatives.h"
namespace sph_native::force_jvp {
// Owner-specific estimator and distance chains. Other force-product owners
// do not include or instantiate the Laplacian's fourth-derivative machinery.
template<class T,int D>struct LaplacianProductPolicy {
    template<class PairType>SPH_HD static void lap_one(const PairType& p,const T(&x)[D],T r,T h,const T(&dt)[D],T dh,T& L,T& dot,T& support){
        L=evaluate_kernel<T,D>(x,h,p.a.kernel).laplacian;if(!p.geometry())return;
        using S=pair_ad::Number<T,false,768>;S xx[D],g[D],hh(h),gh(0);for(int c=0;c<D;++c)xx[c]=S(x[c]);
        warp_compat::laplacian_partials<T,D,false,768>(x,r,h,p.a.kernel,xx,hh,g,gh);
        for(int c=0;c<D;++c)dot+=g[c].v*dt[c];support=gh.v*dh;
    }
    template<class PairType>SPH_HD static void lap_kernel(const PairType& p,const T(&x)[D],T r,T hi,T hj,const T(&dt)[D],T ti,T tj,T& L,T& dL){
        if(p.a.support==15){T li=0,lj=0,di=0,dj=0,si=0,sj=0;lap_one(p,x,r,hi,dt,ti,li,di,si);lap_one(p,x,r,hj,dt,tj,lj,dj,sj);L=(li+lj)*T(.5);dL=(di+si+dj+sj)*T(.5);}
        else{const T h=p.a.support==11?hi:p.a.support==12?hj:p.a.support==13?(hi+hj)/T(2):maximum_support(hi,hj);
            const T dh=p.a.support==11?ti:p.a.support==12?tj:p.a.support==13?(ti+tj)/T(2):hi>=hj?ti:tj;
            T dot=0,support=0;lap_one(p,x,r,h,dt,dh,L,dot,support);dL=dot+support;}
    }
    template<class PairType>SPH_HD static void reverse_lap_kernel(const PairType& p,const T(&x)[D],T r,T hi,T hj,const T(&dt)[D],T ti,T tj,T bL,T bdL,Bars<T,D>& bars,explicit_ad::Trace<T,768>& trace){
        T bx[D]={},bt[D]={},bhi=0,bhj=0,bti=0,btj=0;
        auto one=[&](T h,T dh,T weight,T& bh,T& bth){
            if(p.geometry())warp_compat::laplacian_product_pullback<T,D>(x,r,h,p.a.kernel,dt,dh,bL*weight,bdL*weight,bx,bh,bt,bth,trace);
            else{laplacian_pullback<T,D>(x,r,h,p.a.kernel,bL*weight,bx,bh);warp_compat::laplacian_clipped_replay<T,D>(x,r,h,p.a.kernel,bx,bh,trace);}
        };
        if(p.a.support==15){one(hj,tj,T(.5),bhj,btj);one(hi,ti,T(.5),bhi,bti);}
        else{const T h=p.a.support==11?hi:p.a.support==12?hj:p.a.support==13?(hi+hj)/T(2):maximum_support(hi,hj);
            const T dh=p.a.support==11?ti:p.a.support==12?tj:p.a.support==13?(ti+tj)/T(2):hi>=hj?ti:tj;
            T bh=0,bth=0;one(h,dh,T(1),bh,bth);
            if(p.a.support==11){bhi=bh;bti=bth;}else if(p.a.support==12){bhj=bh;btj=bth;}else if(p.a.support==13){bhi=bhj=bh/T(2);bti=btj=bth/T(2);}
            else{if(maximum_support_query(hi,hj))bhi=bh;else bhj=bh;if(hi>=hj)bti=bth;else btj=bth;}}
        for(int c=0;c<D;++c){bars.add(0,c,p.a.fields,bx[c]);bars.add(1,c,p.a.fields,-bx[c]);bars.add(26,c,p.a.fields,bt[c]);bars.add(27,c,p.a.fields,-bt[c]);}
        bars.add(2,0,p.a.fields,bhi);bars.add(3,0,p.a.fields,bhj);bars.add(28,0,p.a.fields,bti);bars.add(29,0,p.a.fields,btj);
    }

    template<class PairType>SPH_HD static void kernel(const PairType& p,const T(&x)[D],T r,T hi,T hj,const T(&dt)[D],T ti,T tj,T& w,T& dw,T(&g)[D],T(&dg)[D],T& lp,T& dlp){
        if(p.a.laplacian_mode==1)lap_kernel(p,x,r,hi,hj,dt,ti,tj,lp,dlp);else p.kernel(x,r,hi,hj,dt,ti,tj,w,dw,g,dg);
    }
    template<class PairType>SPH_HD static void backward(const PairType& p,const T(&x)[D],T r,T hi,T hj,const T(&dt)[D],T ti,T tj,T bw,T bdw,const T(&bg)[D],const T(&bdg)[D],T bl,T bdl,Bars<T,D>& bars,LaplacianTraceStorage<T>& storage){
        if(p.a.laplacian_mode==1){
            auto* trace=new(&storage.laplacian)explicit_ad::Trace<T,768>;
            reverse_lap_kernel(p,x,r,hi,hj,dt,ti,tj,bl,bdl,bars,*trace);
        }else{
            auto* trace=new(&storage.force)explicit_ad::Trace<T,512>;
            p.template reverse_kernel<true>(x,r,hi,hj,dt,ti,tj,bw,bdw,bg,bdg,bars,trace);
        }
    }
    template<bool Reverse,int C,class PairType,class Read,class Raw>SPH_HD static void field_only(const PairType&p,Read read,Raw raw,const T(&x)[D],T radius,T hi,T hj,T lp,
            pair_ad::Number<T,Reverse,C>B,pair_ad::Number<T,Reverse,C>omega,const pair_ad::Number<T,Reverse,C>(&G)[D],int64_t i,T* sum){
        using S=pair_ad::Number<T,Reverse,C>;const S qi=read(25,0,1);S coeff[9];
        const int fields=p.a.positive_divergence?D:p.a.fields;
        for(int f=0;f<fields;++f)coeff[f]=(explicit_ad::divide(read(40,f),omega)-explicit_ad::divide(read(39,f),qi))*B;
        auto emit=[&](int f,S value){if constexpr(Reverse)value.seed(p.cot[i*p.a.fields+f]);else sum[f]+=value.v;};
        if(p.a.laplacian_mode==1){const S L=raw(lp,56,0);for(int f=0;f<fields;++f)emit(f,coeff[f]*L);return;}
        if(radius<=T(0))return;
        S xx[D],squared(0);for(int c=0;c<D;++c){xx[c]=read(0,c)-read(1,c);xx[c].v=x[c];squared+=xx[c]*xx[c];}
        S rr=pair_ad::sqrt(squared);rr.v=radius;
        const S hi_value=read(2,0),hj_value=read(3,0);
        const S h=p.a.support==11?hi_value:p.a.support==12?hj_value:p.a.support==13?(hi_value+hj_value)/S(2):maximum_support(hi_value,hj_value);
        const S den=rr+S(1e-8)*h;S normal[D];for(int c=0;c<D;++c)normal[c]=explicit_ad::matrix_divide(xx[c],den);
        S projection(0);
        if(p.a.laplacian_mode==4){const S second=rr+S(1e-12)*h;for(int c=0;c<D;++c)projection+=explicit_ad::matrix_divide(normal[c],second)*G[c];}
        else{for(int c=0;c<D;++c)projection+=G[c]*normal[c];projection=explicit_ad::divide(projection,den);}
        if(p.a.laplacian_mode==3){for(int base=0;base<fields;base+=D){S along(0);for(int c=0;c<D;++c)along+=coeff[base+c]*normal[c];
            for(int c=0;c<D;++c)emit(base+c,(coeff[base+c]-S(D+2)*along*normal[c])*projection);}}
        else for(int f=0;f<fields;++f)emit(f,S(-2)*coeff[f]*projection);
    }

    template<bool Reverse,int C,class PairType,class Read,class Raw>SPH_HD static void field(const PairType& p,Read read,Raw raw,const T(&x)[D],T radius,T hi,T hj,T lp,T dlp,
        pair_ad::Number<T,Reverse,C> B,pair_ad::Number<T,Reverse,C> dB,pair_ad::Number<T,Reverse,C> omega,
        const pair_ad::Number<T,Reverse,C>(&G)[D],const pair_ad::Number<T,Reverse,C>(&dG)[D],int64_t i,int64_t j,T* sum){
        using S=pair_ad::Number<T,Reverse,C>;
            if(p.only_fields){field_only<Reverse,C>(p,read,raw,x,radius,hi,hj,lp,B,omega,G,i,sum);return;}
            const S L=raw(lp,56,0),dL=raw(dlp,57,0),qomega=read(25,0,1);
            S q[9],dq[9];const int fields=p.a.positive_divergence?D:p.a.fields;
            // Default computes the field weights before its self-pair guard.
            const bool read_fields=p.a.laplacian_mode==1||p.a.laplacian_mode==4||radius>0||p.a.positive_divergence;
            if(read_fields)for(int f=0;f<fields;++f){const S fi=explicit_ad::divide(read(13,f),qomega),fj=explicit_ad::divide(read(14,f),omega);
                const S dfi=explicit_ad::divide(read(39,f),qomega),dfj=explicit_ad::divide(read(40,f),omega);
                q[f]=(fj-fi)*B;dq[f]=(fj-fi)*dB+(dfj-dfi)*B;}
            S xx[D],dd[D];for(int c=0;c<D;++c){xx[c]=read(0,c)-read(1,c);xx[c].v=x[c];dd[c]=read(26,c)-read(27,c);if(p.bounds_ad)dd[c].v=p.relative_tangent(i,j,c);}
            bool allowed=true;if(p.a.positive_divergence){
                if(p.gate_query&&p.gate_reference){T gate=0;for(int c=0;c<D;++c)
                    gate+=x[c]*((p.gate_reference[j*p.a.fields+c]/omega.v-p.gate_query[i*p.a.fields+c]/qomega.v)*B.v);
                    allowed=gate>=0;
                }else{S gate(0);for(int c=0;c<D;++c)gate+=xx[c]*q[c];allowed=gate.v>=0;}
            }
            auto emit=[&](int f,S result){if constexpr(Reverse)result.seed(p.cot[i*p.a.fields+f]);else sum[f]+=result.v;};
            if(allowed&&p.a.laplacian_mode==1){for(int f=0;f<fields;++f)emit(f,dq[f]*L+q[f]*dL);}
            else if(allowed&&radius>0){
                // This chain uses safe_sqrt(dot(x,x)), whose intermediate
                // squared-norm adjoint can overflow. Do not collapse it into
                // the kernel vectorNorm custom adjoint.
                S squared(0);for(int c=0;c<D;++c)squared+=xx[c]*xx[c];
                S rr=pair_ad::sqrt(squared);rr.v=radius;
                S dr(0);for(int c=0;c<D;++c)dr+=xx[c]*dd[c];dr=explicit_ad::divide(dr,rr);
                const S hqi=read(2,0),hrj=read(3,0),tqi=read(28,0),trj=read(29,0);
                const S h=p.a.support==11?hqi:p.a.support==12?hrj:p.a.support==13?(hqi+hrj)/S(2):maximum_support(hqi,hrj);
                const S dh=p.a.support==11?tqi:p.a.support==12?trj:p.a.support==13?(tqi+trj)/S(2):hi>=hj?tqi:trj;
                const S den=rr+S(1e-8)*h,dden=dr+S(1e-8)*dh;S n[D],dn[D];
                for(int c=0;c<D;++c)n[c]=explicit_ad::matrix_divide(xx[c],den);
                for(int c=0;c<D;++c)dn[c]=explicit_ad::matrix_divide(dd[c]-n[c]*dden,den);
                S P(0),dP(0);
                if(p.a.laplacian_mode==4){const S den2=rr+S(1e-12)*h,dden2=dr+S(1e-12)*dh;S n2[D],dn2[D];
                    for(int c=0;c<D;++c)n2[c]=explicit_ad::matrix_divide(n[c],den2);
                    for(int c=0;c<D;++c)dn2[c]=explicit_ad::matrix_divide(dn[c]-n2[c]*dden2,den2);
                    S first(0),second(0);for(int c=0;c<D;++c)P+=n2[c]*G[c];for(int c=0;c<D;++c)first+=dn2[c]*G[c];for(int c=0;c<D;++c)second+=n2[c]*dG[c];dP=first+second;
                }else{S dot(0),first(0),second(0);for(int c=0;c<D;++c)dot+=G[c]*n[c];for(int c=0;c<D;++c)first+=dG[c]*n[c];for(int c=0;c<D;++c)second+=G[c]*dn[c];
                    P=explicit_ad::divide(dot,den);dP=explicit_ad::divide(first+second,den)-explicit_ad::divide(dot*dden,den*den);}
                if(p.a.laplacian_mode==3){for(int base=0;base<fields;base+=D){
                    // Warp calls its projection helper separately for every
                    // output component. Keep those reverse paths separate.
                    for(int c=0;c<D;++c){S projection(0),dprojection(0);
                        for(int e=0;e<D;++e){projection+=q[base+e]*n[e];dprojection+=dq[base+e]*n[e]+q[base+e]*dn[e];}
                        const S left=S(D+2)*projection*n[c],dleft=S(D+2)*(dprojection*n[c]+projection*dn[c]);
                        emit(base+c,-dleft*P-left*dP+dq[base+c]*P+q[base+c]*dP);}
                }}else for(int f=0;f<fields;++f)emit(f,S(-2)*(dq[f]*P+q[f]*dP));
            }
    }
};
}
