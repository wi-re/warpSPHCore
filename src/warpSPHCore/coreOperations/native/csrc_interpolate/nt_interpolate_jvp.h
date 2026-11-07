#pragma once
#include "execution/dispatch.h"
#include "execution/csrView.h"
#include "execution/neighborhoodAdjoint.h"
#include "math/queryGeometry.h"
#include "kernels/explicit_value_derivatives.h"
#include "util/directionality.h"
namespace sph_native::interpolate_jvp {
using neighborhood::Owner;
constexpr int Slots=26;
struct Buffers{OptionalTensors flat,directions;Tensors transpose,outputs;Tensor cotangent;};
template<int D>struct Layout{
    static constexpr int Q=2*D+2,R=D+D*D+4;
    SPH_HD static bool query(int k){k%=10;return k==0||k==2||k==8||k==9;}
    SPH_HD static int offset(int k){switch(k%10){case 0:case 1:return 0;case 2:case 3:return D;case 4:case 8:return D+1;case 5:case 9:return D+2;case 6:return D+3;default:return D+3+D*D;}}
    SPH_HD static int width(int k,int fields){k%=10;return k==0||k==1||k==9?D:k==6?fields:1;}
};
template<class T,int D>struct Bars{
    T q[2*Layout<D>::Q]={},r[2*Layout<D>::R]={};
    SPH_HD void add(int k,int c,T value){
        const bool query=Layout<D>::query(k);const int base=k>=10?(query?Layout<D>::Q:Layout<D>::R):0;
        (query?q:r)[base+Layout<D>::offset(k)+c]+=warp_compat::rounded_scalar_term(value);
    }
};
template<class T,int D>struct Pair{
    RowArgs<T,D> a;const T *v[Slots],*t[Slots],*cot;T* output[20];
    SPH_HD T value(int slot,int64_t i,T fallback=T(0))const{return v[slot]?v[slot][i]:fallback;}
    SPH_HD T tangent(int slot,int64_t i)const{return t[slot]?t[slot][i]:T(0);}
    SPH_HD T kernel(const T (&dx)[D],T r,T hi,T hj,const T (&dt)[D],T ti,T tj,T& w)const{
        if(a.support==14||a.support==15){
            const warp_compat::ValueProductKernel<T,D> l(dx,r,hi,a.kernel),q(dx,r,hj,a.kernel);w=(l.w+q.w)/T(2);
            T dl=0,dr=0;for(int c=0;c<D;++c){dl+=l.g[c]*dt[c];dr+=q.g[c]*dt[c];}
            return ((dl+dr)+l.wh*ti+q.wh*tj)*T(.5);
        }
        const T h=a.support==11?hi:a.support==12?hj:a.support==13?(hi+hj)/T(2):hi>hj?hi:hj;
        const T dh=a.support==11?ti:a.support==12?tj:a.support==13?(ti+tj)/T(2):hi>=hj?ti:tj;
        const warp_compat::ValueProductKernel<T,D> k(dx,r,h,a.kernel);w=k.w;return k.tangent(dt,dh);
    }
    SPH_HD void kernel_reverse(const T (&dx)[D],T r,T hi,T hj,const T (&dt)[D],T ti,T tj,T bw,T bd,Bars<T,D>& bars)const{
        T bx[D]={},bt[D]={},bhi=0,bhj=0,bti=0,btj=0;
        if(a.support==14||a.support==15){
            const warp_compat::ValueProductKernel<T,D> l(dx,r,hi,a.kernel),q(dx,r,hj,a.kernel);
            q.backward(a.kernel,dt,tj,bw/T(2),bd*T(.5),bx,bhj,bt,btj);
            l.backward(a.kernel,dt,ti,bw/T(2),bd*T(.5),bx,bhi,bt,bti);
        }else{
            const T h=a.support==11?hi:a.support==12?hj:a.support==13?(hi+hj)/T(2):hi>hj?hi:hj;
            const T dh=a.support==11?ti:a.support==12?tj:a.support==13?(ti+tj)/T(2):hi>=hj?ti:tj;
            const warp_compat::ValueProductKernel<T,D> k(dx,r,h,a.kernel);T bh=0,bth=0;
            k.backward(a.kernel,dt,dh,bw,bd,bx,bh,bt,bth);
            if(a.support==11){bhi=bh;bti=bth;}else if(a.support==12){bhj=bh;btj=bth;}
            else if(a.support==13){bhi=bhj=bh/T(2);bti=btj=bth/T(2);}
            else{if(hi>hj)bhi=bh;else bhj=bh;if(hi>=hj)bti=bth;else btj=bth;}
        }
        for(int c=0;c<D;++c){bars.add(0,c,bx[c]);bars.add(1,c,-bx[c]);bars.add(10,c,bt[c]);bars.add(11,c,-bt[c]);}
        bars.add(2,0,bhi);bars.add(3,0,bhj);bars.add(12,0,bti);bars.add(13,0,btj);
    }
    SPH_HD Bars<T,D> interact(int64_t i,int64_t j,T* sum,bool reverse)const{
        Bars<T,D> bars{};if(!query_allowed(a.qkind[i],a.direction)||!reference_allowed(a.rkind[j],a.direction))return bars;
        T dx[D],dt[D];const T r=QueryGeometry<T,D>(a.qx+i*D,a.lo,a.hi,a.periodic).distance(a.rx+j*D,dx);
        for(int c=0;c<D;++c)dt[c]=tangent(0,i*D+c)-tangent(1,j*D+c);
        const T hi=a.qh[i],hj=a.rh[j],ti=tangent(2,i),tj=tangent(3,j);
        T w=0;const T dw=kernel(dx,r,hi,hj,dt,ti,tj,w);
        const bool crk=v[19],volume=v[18];const T A=value(19,i,T(1)),dA=tangent(19,i);
        T poly=1,dpoly=0;
        if(crk)for(int c=0;c<D;++c){poly+=value(20,i*D+c)*dx[c];dpoly+=tangent(20,i*D+c)*dx[c]+value(20,i*D+c)*dt[c];}
        const T C=crk?A*poly:T(1),dC=crk?dA*poly+A*dpoly:T(0);
        const T W=crk?C*w:w,dW=crk?dC*w+C*dw:dw;
        const T mass=value(4,j),rho=value(15,j,T(1)),tm=tangent(4,j),trho=tangent(15,j);
        const T V=volume?value(18,j):mass/rho,dV=volume?tangent(18,j):tm/rho-mass*trho/(rho*rho);
        const T u=dV*W+V*dW,z=V*W;
        T bW=0,bdW=0,bV=0,bdV=0;
        for(int c=0;c<a.fields;++c){
            const auto at=j*a.fields+c;const T f=value(14,at),df=tangent(14,at);
            if(!reverse){sum[c]+=f*u+df*z;continue;}
            const T incoming=cot[i*a.fields+c],bu=f*incoming,bz=df*incoming;
            bars.add(6,c,u*incoming);bars.add(16,c,z*incoming);
            // Match the taped reverse order: field-direction term precedes
            // the primal-field term. Its FMA operand order is observable
            // when singular support bars nearly cancel.
            bV+=warp_compat::rounded_scalar_term(W*bz);bV+=warp_compat::rounded_scalar_term(dW*bu);
            bdV+=warp_compat::rounded_scalar_term(W*bu);
            bW+=warp_compat::rounded_scalar_term(V*bz);bW+=warp_compat::rounded_scalar_term(dV*bu);
            bdW+=warp_compat::rounded_scalar_term(V*bu);
        }
        if(!reverse)return bars;
        if(volume){bars.add(7,0,bV);bars.add(17,0,bdV);}
        else{
            const T n=mass*trho,den=rho*rho,quot=n/den;
            const T bn=-bdV/den,bden=bdV*quot/den;
            bars.add(4,0,trho*bn);bars.add(15,0,mass*bn);
            bars.add(5,0,rho*bden);bars.add(5,0,rho*bden);
            bars.add(14,0,bdV/rho);bars.add(5,0,-bdV*(tm/rho)/rho);
            bars.add(4,0,bV/rho);bars.add(5,0,-bV*V/rho);
        }
        T bw=bW,bd=bdW;
        if(crk){
            const T bC=w*bW+dw*bdW,bdC=w*bdW;
            bw=C*bW+dC*bdW;bd=C*bdW;
            bars.add(8,0,poly*bC+dpoly*bdC);bars.add(18,0,poly*bdC);
            const T bp=A*bC+dA*bdC,bdp=A*bdC;
            for(int c=0;c<D;++c){
                bars.add(9,c,dx[c]*bp+dt[c]*bdp);bars.add(19,c,dx[c]*bdp);
                const T bx=value(20,i*D+c)*bp+tangent(20,i*D+c)*bdp,bt=value(20,i*D+c)*bdp;
                bars.add(0,c,bx);bars.add(1,c,-bx);bars.add(10,c,bt);bars.add(11,c,-bt);
            }
        }
        kernel_reverse(dx,r,hi,hj,dt,ti,tj,bw,bd,bars);return bars;
    }
};
template<class T,int D>struct Pullback{
    Pair<T,D> p;
    template<Owner Side>SPH_HD int64_t prepare(int64_t i)const{return i;}
    template<Owner Side>SPH_HD auto identity()const{return neighborhood::Sums<T,2*(Side==Owner::Query?Layout<D>::Q:Layout<D>::R)>{};}
    template<Owner Side,class State>SPH_HD void interact(int64_t owner,int64_t,neighborhood::Neighbor neighbor,State& sum)const{
        const auto bars=p.interact(Side==Owner::Query?owner:neighbor.reference,Side==Owner::Query?neighbor.reference:owner,nullptr,true);
        for(int c=0;c<2*(Side==Owner::Query?Layout<D>::Q:Layout<D>::R);++c)sum.values[c]+=(Side==Owner::Query?bars.q:bars.r)[c];
    }
    template<Owner Side,class State>SPH_HD void finish(int64_t owner,const State& sum)const{
        for(int k=0;k<20;++k)if(Layout<D>::query(k)==(Side==Owner::Query)){
            const int base=k>=10?(Side==Owner::Query?Layout<D>::Q:Layout<D>::R):0;
            const int width=Layout<D>::width(k,p.a.fields),at=base+Layout<D>::offset(k);
            for(int c=0;c<width;++c)p.output[k][owner*width+c]=sum.values[at+c];
        }
    }
};
template<class T,int D>struct ProductRow{
    Pair<T,D> p;
    SPH_HD void operator()(int64_t i)const{
        T sum[9]={};for(int64_t k=0;k<p.a.counts[i];++k)p.interact(i,p.a.neighbors[p.a.offsets[i]+k],sum,false);
        for(int c=0;c<p.a.fields;++c)p.a.output[i*p.a.fields+c]=sum[c];
    }
};
struct Rows{
    const Buffers& b;
    template<class T,int D,class Launch>void operator()(int64_t n,RowArgs<T,D> a,const Launch& launch)const{
        Pair<T,D> p{a,{}, {},b.cotangent.defined()?ptr<T>(b.cotangent):nullptr,{}};
        for(int k=0;k<Slots;++k){
            p.v[k]=b.flat[k]&&(b.flat[k]->scalar_type()==ScalarType::Float||b.flat[k]->scalar_type()==ScalarType::Double)?ptr<T>(*b.flat[k]):nullptr;
            p.t[k]=b.directions[k]?ptr<T>(*b.directions[k]):nullptr;
        }
        if(!b.cotangent.defined()){launch(n,ProductRow<T,D>{p});return;}
        for(int k=0;k<20;++k)p.output[k]=b.outputs[k].mutable_data_ptr<T>();
        const Pullback<T,D> pull{p};const neighborhood::CsrView csr{a.offsets,a.counts,a.neighbors};
        launch(n,neighborhood::OwnedAdjointRow<Owner::Query,decltype(pull),decltype(csr)>{pull,csr});
        const auto& t=b.transpose;const neighborhood::CsrView csc{index_view(t[0]),ptr<int32_t>(t[1]),index_view(t[2])};
        launch(a.references,neighborhood::OwnedAdjointRow<Owner::Reference,decltype(pull),decltype(csc)>{pull,csc});
    }
};
}
