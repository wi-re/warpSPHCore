#pragma once
#include "execution/explicitForceProduct.h"
namespace sph_native::force_jvp {
template<class T,int D>struct CovarianceProductPolicy {
    template<bool Reverse,int C,class P,class Read>SPH_HD static void field(
            const P&p,Read read,const T(&x)[D],pair_ad::Number<T,Reverse,C> V,pair_ad::Number<T,Reverse,C> dV,
            const pair_ad::Number<T,Reverse,C>(&G)[D],const pair_ad::Number<T,Reverse,C>(&dG)[D],int64_t i,T* sum){
        using S=pair_ad::Number<T,Reverse,C>;
        S y[D],dy[D],first[D*D],second[D*D],third[D*D];
        for(int c=0;c<D;++c){y[c]=read(1,c)-read(0,c);y[c].v=-x[c];dy[c]=read(27,c)-read(26,c);}
        // Matrix statements stay separate. Reverse creation order makes each
        // outer-product pullback reduce columns/rows in ascending order.
        for(int c=D-1;c>=0;--c)for(int e=D-1;e>=0;--e)first[c*D+e]=dV*(y[c]*G[e]);
        for(int c=D-1;c>=0;--c)for(int e=D-1;e>=0;--e)second[c*D+e]=V*(dy[c]*G[e]);
        for(int c=D-1;c>=0;--c)for(int e=D-1;e>=0;--e)third[c*D+e]=V*(y[c]*dG[e]);
        for(int c=0;c<D*D;++c){const S result=first[c]+second[c]+third[c];
            if constexpr(Reverse)result.seed(p.cot[i*D*D+c]);else sum[c]+=result.v;}
    }
};
}
