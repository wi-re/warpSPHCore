#pragma once
#include "execution/dispatch.h"
#include "execution/csrView.h"
#include "execution/neighborhoodAdjoint.h"
#include "math/queryGeometry.h"
#include "kernels/explicit_force_derivatives.h"
#include "util/directionality.h"
#include <new>
namespace sph_native::force_jvp {
using neighborhood::Owner;
template<class T,int D>struct LaplacianProductPolicy;
template<class T,int D>struct CovarianceProductPolicy;
constexpr int Slots=26,Capacity=1536;
// The Laplacian field trace is consumed before its kernel pullback. Reuse its
// storage for those later traces instead of stacking all estimator branches.
template<class T>union LaplacianTraceStorage {
    explicit_ad::Trace<T,Capacity> pair;
    explicit_ad::Trace<T,768> laplacian;
    explicit_ad::Trace<T,512> force;
    SPH_HD LaplacianTraceStorage(){}
};
struct Buffers{OptionalTensors flat,directions;Tensors transpose,outputs;Tensor cotangent;bool consistent=false,bounds_ad=false;OptionalTensors gates;};
template<int D>struct Layout{
    SPH_HD static bool query(int k){k%=Slots;return !(k==1||k==3||k==4||k==14||k==15||k==18||k==23);}
    SPH_HD static int width(int k,int f){k%=Slots;switch(k){case 0:case 1:case 7:case 8:case 20:case 21:return D;
        case 22:case 24:return D*D;case 13:case 14:return f;
        case 2:case 3:case 4:case 15:case 16:case 17:case 18:case 19:case 23:case 25:return 1;default:return 0;}}
    SPH_HD static int offset(int k,int f){const bool owner=query(k);int at=0;for(int c=0;c<k;++c)if(query(c)==owner)at+=width(c,f);return at;}
    static constexpr int Q=5*D+3*D*D+5,R=D+D*D+5;
};
template<class T,int D>struct Bars{T q[2*Layout<D>::Q]={},r[2*Layout<D>::R]={};
    SPH_HD void add(int slot,int at,int fields,T value){(Layout<D>::query(slot)?q:r)[Layout<D>::offset(slot,fields)+at]+=value;}
};
template<class T,int D,int Operation>struct Pair{
    RowArgs<T,D> a;const T *v[Slots],*t[Slots],*cot;T* output[52];bool consistent=false,bounds_ad=false,only_fields=false;const T* gate_query=nullptr;const T* gate_reference=nullptr;
    SPH_HD bool geometry()const{return t[0]||t[1]||t[2]||t[3]||(bounds_ad&&(t[7]||t[8]));}
    SPH_HD bool positive_pair(int64_t i,int64_t j,const T(&x)[D])const{
        const T m=value(4,j),rhoi=value(16,i,T(1)),rhoj=value(15,j,T(1));
        const T weight=a.scheme==2?m*rhoi/(rhoj*rhoj):v[18]?value(18,j):m/rhoj;
        const T qi=value(25,i,T(1)),rj=value(23,j,T(1));T gate=0;
        for(int c=0;c<D;++c){const T fi=(gate_query?gate_query[i*a.fields+c]:value(13,i*a.fields+c))/qi;
            const T fj=(gate_reference?gate_reference[j*a.fields+c]:value(14,j*a.fields+c))/rj;
            gate+=x[c]*((fj-fi)*weight);}
        return gate>=0;
    }
    SPH_HD T winding(int64_t i,int64_t j,int c)const{
        if(!a.periodic[c])return T(0);
        const T length=a.hi[c]-a.lo[c],half=length/T(2),delta=a.qx[i*D+c]-a.rx[j*D+c];
        if(delta>=-half&&delta<half)return T(0);
        if(delta>=half&&delta<length)return T(1);
        if(delta<-half&&delta>=-length)return T(-1);
        return scalar_floor((delta+half)/length);
    }
    SPH_HD T relative_tangent(int64_t i,int64_t j,int c)const{
        T result=tangent(0,i*D+c)-tangent(1,j*D+c);
        if(bounds_ad){const T turns=winding(i,j,c);if(turns!=T(0))result+=turns*(tangent(7,c)-tangent(8,c));}
        return result;
    }
    SPH_HD T value(int slot,int64_t at,T fallback=T(0))const{return v[slot]?v[slot][at]:fallback;}
    SPH_HD T tangent(int slot,int64_t at)const{return t[slot]?t[slot][at]:T(0);}
    SPH_HD void gradient_value(const T(&x)[D],T radius,T h,T(&g)[D])const{
        if(h<=T(0)||radius>h)return;
        const T eps=sizeof(T)==4?T(1e-7):T(1e-15),den=radius>eps?radius:eps;
        const T shape=warp_compat::gradient_shape_derivative<T,D>(radius/h,a.kernel,radial_shape<T,D>(radius/h,a.kernel).d1);
        const T term=shape*(normalization<T,D>(a.kernel)/power<T,D>(h,D+1));
        for(int c=0;c<D;++c)g[c]=(x[c]/den)*term;
    }
    SPH_HD void kernel_one(const T(&x)[D],T radius,T h,const T(&dt)[D],T dh,T& w,T& dw,T(&g)[D],T(&dg)[D])const{
        const warp_compat::ValueProductKernel<T,D> k(x,radius,h,a.kernel);w=k.w;gradient_value(x,radius,h,g);
        if(!geometry()){dw=0;return;}
        T dot=0;for(int c=0;c<D;++c)dot+=g[c]*dt[c];dw=dot+k.wh*dh;
        for(int c=0;c<D;++c){T product=0;for(int e=0;e<D;++e)product+=k.H[c*D+e]*dt[e];dg[c]=product+k.gh[c]*dh;}
    }
    SPH_HD void kernel(const T(&x)[D],T radius,T hi,T hj,const T(&dt)[D],T ti,T tj,T& w,T& dw,T(&g)[D],T(&dg)[D])const{
        if(a.support==14||a.support==15){
            const warp_compat::ValueProductKernel<T,D> ki(x,radius,hi,a.kernel),kj(x,radius,hj,a.kernel);T gi[D]={},gj[D]={},di=0,dj=0;
            gradient_value(x,radius,hi,gi);gradient_value(x,radius,hj,gj);
            if(!geometry()){w=(ki.w+kj.w)*T(.5);dw=0;for(int c=0;c<D;++c)g[c]=(gi[c]+gj[c])*T(.5);return;}
            for(int c=0;c<D;++c){di+=gi[c]*dt[c];dj+=gj[c]*dt[c];}
            w=(ki.w+kj.w)*T(.5);dw=((di+dj)+ki.wh*ti+kj.wh*tj)*T(.5);
            for(int c=0;c<D;++c){T hdi=0,hdj=0;for(int e=0;e<D;++e){hdi+=ki.H[c*D+e]*dt[e];hdj+=kj.H[c*D+e]*dt[e];}
                g[c]=(gi[c]+gj[c])*T(.5);dg[c]=((hdi+hdj)+ki.gh[c]*ti+kj.gh[c]*tj)*T(.5);}
        }else{const T h=a.support==11?hi:a.support==12?hj:a.support==13?(hi+hj)/T(2):hi>hj?hi:hj;
            const T dh=a.support==11?ti:a.support==12?tj:a.support==13?(ti+tj)/T(2):hi>=hj?ti:tj;
            kernel_one(x,radius,h,dt,dh,w,dw,g,dg);}
    }
    template<bool Borrow=false>SPH_HD void reverse_one(const T(&x)[D],T radius,T h,const T(&dt)[D],T dh,T bw,T bdw,const T(&bg)[D],const T(&bdg)[D],
            T(&bx)[D],T& bh,T(&bt)[D],T& bth,explicit_ad::Trace<T,512>* scratch=nullptr)const{
        if(!geometry()){
            warp_compat::force_pullback<T,D>(x,radius,h,a.kernel,T(0),bg,bx,bh);
            if(v[19])warp_compat::value_pullback<T,D>(x,radius,h,a.kernel,bw,bx,bh);
            return;
        }
        if constexpr(Borrow)warp_compat::force_product_pullback<T,D>(x,radius,h,a.kernel,dt,dh,bg,bdg,bx,bh,bt,bth,*scratch);
        else warp_compat::force_product_pullback<T,D>(x,radius,h,a.kernel,dt,dh,bg,bdg,bx,bh,bt,bth);
        if(v[19]){const warp_compat::ValueProductKernel<T,D> k(x,radius,h,a.kernel);k.backward(a.kernel,dt,dh,bw,bdw,bx,bh,bt,bth);}
    }
    template<bool Borrow=false>SPH_HD void reverse_kernel(const T(&x)[D],T radius,T hi,T hj,const T(&dt)[D],T ti,T tj,T bw,T bdw,const T(&bg)[D],const T(&bdg)[D],Bars<T,D>& bars,explicit_ad::Trace<T,512>* scratch=nullptr)const{
        T bx[D]={},bt[D]={},bhi=0,bhj=0,bti=0,btj=0;
        if(a.support==14||a.support==15){T bg2[D],bdg2[D];for(int c=0;c<D;++c){bg2[c]=bg[c]*T(.5);bdg2[c]=bdg[c]*T(.5);}
            reverse_one<Borrow>(x,radius,hj,dt,tj,bw*T(.5),bdw*T(.5),bg2,bdg2,bx,bhj,bt,btj,scratch);
            reverse_one<Borrow>(x,radius,hi,dt,ti,bw*T(.5),bdw*T(.5),bg2,bdg2,bx,bhi,bt,bti,scratch);
        }else{const T h=a.support==11?hi:a.support==12?hj:a.support==13?(hi+hj)/T(2):hi>hj?hi:hj;
            const T dh=a.support==11?ti:a.support==12?tj:a.support==13?(ti+tj)/T(2):hi>=hj?ti:tj;
            T bh=0,bth=0;reverse_one<Borrow>(x,radius,h,dt,dh,bw,bdw,bg,bdg,bx,bh,bt,bth,scratch);
            if(a.support==11){bhi=bh;bti=bth;}else if(a.support==12){bhj=bh;btj=bth;}else if(a.support==13){bhi=bhj=bh/T(2);bti=btj=bth/T(2);}
            else{if(hi>hj)bhi=bh;else bhj=bh;if(hi>=hj)bti=bth;else btj=bth;}}
        for(int c=0;c<D;++c){bars.add(0,c,a.fields,bx[c]);bars.add(1,c,a.fields,-bx[c]);bars.add(26,c,a.fields,bt[c]);bars.add(27,c,a.fields,-bt[c]);}
        bars.add(2,0,a.fields,bhi);bars.add(3,0,a.fields,bhj);bars.add(28,0,a.fields,bti);bars.add(29,0,a.fields,btj);
    }
    template<bool Reverse>SPH_HD Bars<T,D> interact(int64_t i,int64_t j,T* sum)const{
        Bars<T,D> bars{};if(!query_allowed(a.qkind[i],a.direction)||!reference_allowed(a.rkind[j],a.direction))return bars;
        if constexpr(Operation==4){if constexpr(D==1)return bars;else if(a.fields<D)return bars;}
        T x[D],dt[D];const T radius=QueryGeometry<T,D>(a.qx+i*D,a.lo,a.hi,a.periodic).distance(a.rx+j*D,x);
        if constexpr(Operation==5)if(bounds_ad&&a.positive_divergence&&(only_fields||(gate_query&&gate_reference))&&!positive_pair(i,j,x))return bars;
        for(int c=0;c<D;++c)dt[c]=relative_tangent(i,j,c);
        const T hi=a.qh[i],hj=a.rh[j],ti=tangent(2,i),tj=tangent(3,j);
        T w=0,dw=0,g[D]={},dg[D]={},lp=0,dlp=0;
        if constexpr(Operation==5)LaplacianProductPolicy<T,D>::kernel(*this,x,radius,hi,hj,dt,ti,tj,w,dw,g,dg,lp,dlp);
        else kernel(x,radius,hi,hj,dt,ti,tj,w,dw,g,dg);
        using S=pair_ad::Number<T,Reverse,Capacity>;using Trace=explicit_ad::Trace<T,Capacity>;
        using ReverseStorage=std::conditional_t<Operation==5,LaplacianTraceStorage<T>,Trace>;
        using Storage=std::conditional_t<Reverse,ReverseStorage,char>;Storage storage;Trace* trace=nullptr;
        if constexpr(Reverse){if constexpr(Operation==5)trace=new(&storage.pair)Trace;else trace=&storage;}
        auto read=[&](int slot,int c,T fallback=T(0)){
            const int k=slot%Slots;const bool direction=slot>=Slots;const int width=Layout<D>::width(k,a.fields);
            const int64_t at=(Layout<D>::query(k)?i:j)*width+c;const T* data=direction?t[k]:v[k];
            return S::leaf(data?data[at]:fallback,0,trace,slot,c,Reverse&&data);};
        auto raw=[&](T number,int slot,int c){const bool active=(slot!=53&&slot!=55&&slot!=57)||geometry();return S::leaf(number,0,trace,slot,c,Reverse&&active);};
        S W=raw(w,52,0),dW=raw(dw,53,0),G[D],dG[D];for(int c=0;c<D;++c){G[c]=raw(g[c],54,c);dG[c]=raw(dg[c],55,c);}
        if(v[19]&&Operation==5&&only_fields){
            const S A=read(19,0);S poly(1),xx[D],B[D],ga[D],gb[D*D];
            for(int c=0;c<D;++c){B[c]=read(20,c);ga[c]=read(21,c);xx[c]=read(0,c)-read(1,c);xx[c].v=x[c];poly+=B[c]*xx[c];}
            for(int c=0;c<D*D;++c)gb[c]=read(22,c);
            for(int c=0;c<D;++c){S product(0);for(int e=0;e<D;++e)product+=gb[c*D+e]*xx[e];
                G[c]=(A*W)*B[c]+(A*poly)*G[c]+(poly*W)*ga[c]+(A*W)*product;}
        }else if(v[19]){
            const S A=read(19,0),dA=read(45,0);S B[D],dB[D],ga[D],dga[D],xx[D],dd[D],gb[D*D],dgb[D*D],poly(1),dp1(0),dp2(0);
            for(int c=0;c<D;++c){B[c]=read(20,c);dB[c]=read(46,c);ga[c]=read(21,c);dga[c]=read(47,c);
                xx[c]=read(0,c)-read(1,c);xx[c].v=x[c];dd[c]=read(26,c)-read(27,c);if(bounds_ad)dd[c].v=dt[c];
                poly+=B[c]*xx[c];dp1+=dB[c]*xx[c];dp2+=B[c]*dd[c];}
            for(int c=0;c<D*D;++c){gb[c]=read(22,c);dgb[c]=read(48,c);}
            const S dp=dp1+dp2;
            S out[D],dout[D],term1[D],dterm1[D],term2[D],dterm2[D],term3[D],dterm3[D],term4[D],dterm4[D];
            const S f1=A*W,df1a=dA*W,df1b=A*dW,df1=df1a+df1b;
            for(int c=0;c<D;++c){term1[c]=f1*B[c];const S a= df1*B[c],b=f1*dB[c];dterm1[c]=a+b;}
            const S f2=A*poly,df2a=dA*poly,df2b=A*dp,df2=df2a+df2b;
            for(int c=0;c<D;++c){term2[c]=f2*G[c];const S a=df2*G[c],b=f2*dG[c];dterm2[c]=a+b;}
            const S f3=poly*W,df3a=dp*W,df3b=poly*dW,df3=df3a+df3b;
            for(int c=0;c<D;++c){term3[c]=f3*ga[c];const S a=df3*ga[c],b=f3*dga[c];dterm3[c]=a+b;}
            const S f4=A*W,df4a=dA*W,df4b=A*dW,df4=df4a+df4b;
            for(int c=0;c<D;++c){S product(0),dproduct1(0),dproduct2(0);for(int e=0;e<D;++e){product+=gb[c*D+e]*xx[e];dproduct1+=dgb[c*D+e]*xx[e];dproduct2+=gb[c*D+e]*dd[e];}
                const S dproduct=dproduct1+dproduct2;term4[c]=f4*product;const S a=df4*product,b=f4*dproduct;dterm4[c]=a+b;}
            // Keep the vector statements in the reference's tape order.
            for(int c=0;c<D;++c)out[c]=term1[c]+term2[c]+term3[c]+term4[c];
            for(int c=0;c<D;++c)dout[c]=dterm1[c]+dterm2[c]+dterm3[c]+dterm4[c];
            for(int c=0;c<D;++c){G[c]=out[c];dG[c]=dout[c];}
        }
        if(v[24]&&Operation==5&&only_fields){S out[D];for(int c=0;c<D;++c){out[c]=S(0);for(int e=0;e<D;++e)out[c]+=read(24,c*D+e)*G[e];}for(int c=0;c<D;++c)G[c]=out[c];}
        else if(v[24]){S L[D*D],dL[D*D],o[D],dO[D],dO1[D],dO2[D];for(int c=0;c<D*D;++c){L[c]=read(24,c);dL[c]=read(50,c);}
            for(int c=0;c<D;++c){dO1[c]=S(0);for(int e=0;e<D;++e)dO1[c]+=dL[c*D+e]*G[e];}
            for(int c=0;c<D;++c){dO2[c]=S(0);for(int e=0;e<D;++e)dO2[c]+=L[c*D+e]*dG[e];}
            for(int c=0;c<D;++c)dO[c]=dO1[c]+dO2[c];
            for(int c=0;c<D;++c){o[c]=S(0);for(int e=0;e<D;++e)o[c]+=L[c*D+e]*G[e];}
            for(int c=0;c<D;++c){G[c]=o[c];dG[c]=dO[c];}}
        const S m=read(4,0),rhoi=read(16,0,1),rhoj=read(15,0,1),tm=read(30,0),tri=read(42,0),trj=read(41,0);
        S V(0),dV(0),A(0),B(0),dA(0),dB(0);
        if(Operation==5&&only_fields){B=a.scheme==2?explicit_ad::divide(m*rhoi,rhoj*rhoj):v[18]?read(18,0):explicit_ad::divide(m,rhoj);}
        else{

        if constexpr(Operation==3){if(consistent&&a.scheme!=2){
            if(v[18]){const S volume=read(18,0),dv=read(44,0);V=explicit_ad::divide(volume*rhoj,rhoi);
                dV=explicit_ad::divide(dv*rhoj+volume*trj,rhoi)-explicit_ad::divide(volume*rhoj*tri,rhoi*rhoi);}
            else{V=explicit_ad::divide(m,rhoi);dV=explicit_ad::divide(tm,rhoi)-explicit_ad::divide(m*tri,rhoi*rhoi);}
        }else{
            V=v[18]?read(18,0):explicit_ad::divide(m,rhoj);
            dV=v[18]?read(44,0):explicit_ad::divide(tm,rhoj)-explicit_ad::divide(m*trj,rhoj*rhoj);
        }}else{
            V=v[18]?read(18,0):explicit_ad::divide(m,rhoj);
            dV=v[18]?read(44,0):explicit_ad::divide(tm,rhoj)-explicit_ad::divide(m*trj,rhoj*rhoj);
        }
        B=V;dB=dV;
        if constexpr(Operation!=7){
        if(a.scheme==3){A=-V;dA=-dV;}else if(a.scheme==4){A=V;dA=dV;}
        else if(a.scheme==2){A=explicit_ad::divide(m,rhoi);dA=explicit_ad::divide(tm,rhoi)-explicit_ad::divide(m*tri,rhoi*rhoi);
            B=explicit_ad::divide(m*rhoi,rhoj*rhoj);dB=explicit_ad::divide(tm*rhoi,rhoj*rhoj)+explicit_ad::divide(m*tri,rhoj*rhoj)-explicit_ad::divide(S(2)*m*rhoi*trj,rhoj*rhoj*rhoj);}
        }
        }
        const S omega=read(23,0,1);
        if constexpr(Operation==2){
            for(int f=0;f<a.fields;++f){const S fi=read(13,f),fj=explicit_ad::divide(read(14,f),omega),dfi=read(39,f),dfj=explicit_ad::divide(read(40,f),omega);
                const S coeff=fi*A+fj*B,dc=fi*dA+fj*dB,df=dfi*A+dfj*B;
                for(int c=0;c<D;++c){const S result=dc*G[c]+coeff*dG[c]+df*G[c];
                    if constexpr(Reverse)result.seed(cot[(i*a.fields+f)*D+c]);else sum[f*D+c]+=result.v;}}
        }else if constexpr(Operation==3){
            const S qomega=read(25,0,1);const int groups=a.fields/D;
            for(int group=0;group<groups;++group){S dc[D],coeff[D],df[D];
                for(int c=0;c<D;++c){const int f=a.dot_mode?c*groups+group:group*D+c;
                    const S fi=explicit_ad::divide(read(13,f),qomega),fj=explicit_ad::divide(read(14,f),omega);
                    const S dfi=explicit_ad::divide(read(39,f),qomega),dfj=explicit_ad::divide(read(40,f),omega);
                    coeff[c]=fi*A+fj*B;dc[c]=fi*dA+fj*dB;df[c]=dfi*A+dfj*B;}
                S first(0),second(0),third(0);
                for(int c=0;c<D;++c)first+=dc[c]*G[c];
                for(int c=0;c<D;++c)second+=coeff[c]*dG[c];
                for(int c=0;c<D;++c)third+=df[c]*G[c];
                const S result=first+second+third;
                if constexpr(Reverse)result.seed(cot[i*groups+group]);else sum[group]+=result.v;}
        }
        if constexpr(Operation==4&&D==2){
            const S qomega=read(25,0,1);const int groups=a.fields/2;
            for(int group=0;group<groups;++group){S dc[2],coeff[2],df[2];
                for(int c=0;c<2;++c){const int f=c*groups+group;
                    const S fi=explicit_ad::divide(read(13,f),qomega),fj=explicit_ad::divide(read(14,f),omega);
                    const S dfi=explicit_ad::divide(read(39,f),qomega),dfj=explicit_ad::divide(read(40,f),omega);
                    coeff[c]=fi*A+fj*B;dc[c]=fi*dA+fj*dB;df[c]=dfi*A+dfj*B;}
                const S result=G[0]*dc[1]-G[1]*dc[0]+dG[0]*coeff[1]-dG[1]*coeff[0]+G[0]*df[1]-G[1]*df[0];
                if constexpr(Reverse)result.seed(cot[i*groups+group]);else sum[group]+=result.v;
            }
        }
        if constexpr(Operation==7)CovarianceProductPolicy<T,D>::template field<Reverse,Capacity>(*this,read,x,V,dV,G,dG,i,sum);
        if constexpr(Operation==5)LaplacianProductPolicy<T,D>::template field<Reverse,Capacity>(*this,read,raw,x,radius,hi,hj,lp,dlp,B,dB,omega,G,dG,i,j,sum);
        if constexpr(Reverse){trace->reverse();T bw=0,bdw=0,bg[D]={},bdg[D]={},bl=0,bdl=0;
            for(int k=0;k<trace->inputs;++k){const auto leaf=trace->leaves[k];const T bar=trace->nodes[leaf.node].bar;
                if(leaf.slot==52)bw+=bar;else if(leaf.slot==53)bdw+=bar;else if(leaf.slot==54)bg[leaf.at]+=bar;else if(leaf.slot==55)bdg[leaf.at]+=bar;
                else if(leaf.slot==56)bl+=bar;else if(leaf.slot==57)bdl+=bar;else bars.add(leaf.slot,leaf.at,a.fields,bar);}
            if constexpr(Operation==5)LaplacianProductPolicy<T,D>::backward(*this,x,radius,hi,hj,dt,ti,tj,bw,bdw,bg,bdg,bl,bdl,bars,storage);
            else reverse_kernel(x,radius,hi,hj,dt,ti,tj,bw,bdw,bg,bdg,bars);
            if(bounds_ad)for(int c=0;c<D;++c){const T turns=winding(i,j,c);if(turns==T(0))continue;
                const T bx=bars.q[Layout<D>::offset(0,a.fields)+c],bt=bars.q[Layout<D>::offset(26,a.fields)+c];
                bars.add(7,c,a.fields,turns*bx);bars.add(8,c,a.fields,-turns*bx);
                bars.add(33,c,a.fields,turns*bt);bars.add(34,c,a.fields,-turns*bt);
            }}
        return bars;
    }
};
template<class T,int D,int Operation>struct Pullback{Pair<T,D,Operation> p;
    template<Owner Side>SPH_HD int64_t prepare(int64_t i)const{return i;}
    template<Owner Side>SPH_HD auto identity()const{return neighborhood::Sums<T,2*(Side==Owner::Query?Layout<D>::Q:Layout<D>::R)>{};}
    template<Owner Side,class State>SPH_HD void interact(int64_t owner,int64_t,neighborhood::Neighbor n,State& sum)const{
        const auto bars=p.template interact<true>(Side==Owner::Query?owner:n.reference,Side==Owner::Query?n.reference:owner,nullptr);
        for(int c=0;c<2*(Side==Owner::Query?Layout<D>::Q:Layout<D>::R);++c)sum.values[c]+=(Side==Owner::Query?bars.q:bars.r)[c];}
    template<Owner Side,class State>SPH_HD void finish(int64_t owner,const State& sum)const{
        for(int k=0;k<52;++k)if(((k%26!=7&&k%26!=8)||p.bounds_ad)&&Layout<D>::query(k)==(Side==Owner::Query)&&p.output[k]){const int width=Layout<D>::width(k,p.a.fields),at=Layout<D>::offset(k,p.a.fields);
            for(int c=0;c<width;++c)p.output[k][owner*width+c]=sum.values[at+c];}}
};
template<class T,int D,int Operation>struct ProductRow{Pair<T,D,Operation> p;SPH_HD void operator()(int64_t i)const{
    T sum[27]={};for(int64_t k=0;k<p.a.counts[i];++k)p.template interact<false>(i,p.a.neighbors[p.a.offsets[i]+k],sum);
    const int width=Operation==7?D*D:Operation==2?p.a.fields*D:Operation==3?p.a.fields/D:Operation==4?(D==1?1:D==2?p.a.fields/2:p.a.fields):p.a.fields;for(int c=0;c<width;++c)p.a.output[i*width+c]=sum[c];}};
template<int Operation>struct Rows{const Buffers& b;template<class T,int D,class Launch>void operator()(int64_t n,RowArgs<T,D>a,const Launch&launch)const{
    Pair<T,D,Operation>p{a,{}, {},b.cotangent.defined()?ptr<T>(b.cotangent):nullptr,{},b.consistent,b.bounds_ad,false};
    p.only_fields=Operation==5&&b.bounds_ad;for(int k=0;k<Slots;++k)if(k!=13&&k!=14&&b.directions[k])p.only_fields=false;
    if(b.gates.size()==2){p.gate_query=ptr<T>(*b.gates[0]);p.gate_reference=ptr<T>(*b.gates[1]);}
    for(int k=0;k<Slots;++k){p.v[k]=b.flat[k]&&(b.flat[k]->scalar_type()==ScalarType::Float||b.flat[k]->scalar_type()==ScalarType::Double)?ptr<T>(*b.flat[k]):nullptr;p.t[k]=b.directions[k]?ptr<T>(*b.directions[k]):nullptr;}
    if(!b.cotangent.defined()){launch(n,ProductRow<T,D,Operation>{p});return;}
    for(int k=0;k<52;++k)p.output[k]=b.outputs[k].numel()?b.outputs[k].mutable_data_ptr<T>():nullptr;
    const Pullback<T,D,Operation>pull{p};const neighborhood::CsrView csr{a.offsets,a.counts,a.neighbors};launch(n,neighborhood::OwnedAdjointRow<Owner::Query,decltype(pull),decltype(csr)>{pull,csr});
    const auto&t=b.transpose;const neighborhood::CsrView csc{index_view(t[0]),ptr<int32_t>(t[1]),index_view(t[2])};launch(a.references,neighborhood::OwnedAdjointRow<Owner::Reference,decltype(pull),decltype(csc)>{pull,csc});
}};
}
