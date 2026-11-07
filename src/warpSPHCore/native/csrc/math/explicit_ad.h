#pragma once
#include "math/pair_ad.h"
namespace sph_native::explicit_ad {
template<class T>SPH_HD T rounded(T x){
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return x;
#else
    volatile T value=x;return value;
#endif
}
template<class T>SPH_HD void accumulate(T& bar,T a,T b){
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr(sizeof(T)==4)bar=__fmaf_rn(a,b,bar);else bar=__fma_rn(a,b,bar);
#else
    bar+=rounded(a*b);
#endif
}
// Retain the reference quotient pullback's operation order, including overflow.
// Ordinary pair_ad consumers retain their original arithmetic policy.
template<class T,int Capacity>struct Trace:pair_ad::Trace<T,Capacity>{
    unsigned char division[Capacity]={};
    SPH_HD void reverse(){
        for(int n=this->count-1;n>=0;--n){const auto x=this->nodes[n];
            if(division[n]){
                const T q=x.bar/x.db;
                if(x.a>=0)this->nodes[x.a].bar+=q;
                if(x.b>=0)this->nodes[x.b].bar-=division[n]==2 ? x.da*x.bar/(x.db*x.db) : x.bar*x.da/x.db;
            }else{
                if(x.a>=0)accumulate(this->nodes[x.a].bar,x.da,x.bar);
                if(x.b>=0)accumulate(this->nodes[x.b].bar,x.db,x.bar);
            }
        }
    }
};
template<class T,bool Reverse,int C>SPH_HD auto divide(pair_ad::Number<T,Reverse,C> a,pair_ad::Number<T,Reverse,C> b){
    using S=pair_ad::Number<T,Reverse,C>;const T value=a.v/b.v;
    if constexpr(Reverse){
        S result=S::binary(value,a,b,value,b.v);
        if(result.node>=0)static_cast<Trace<T,C>*>(result.trace)->division[result.node]=true;
        return result;
    }else return S::binary(value,a,b,T(1)/b.v,-value/b.v);
}
// Matrix/scalar division tapes a tensor-dot before dividing by s squared.
template<class T,bool Reverse,int C>SPH_HD auto matrix_divide(pair_ad::Number<T,Reverse,C> a,pair_ad::Number<T,Reverse,C> b){
    using S=pair_ad::Number<T,Reverse,C>;const T value=a.v/b.v;
    if constexpr(Reverse){S result=S::binary(value,a,b,a.v,b.v);if(result.node>=0)static_cast<Trace<T,C>*>(result.trace)->division[result.node]=2;return result;}
    else return S::binary(value,a,b,T(1)/b.v,-a.v/(b.v*b.v));
}
template<class S>SPH_HD S power(S x,int n){S result=x;for(int k=1;k<n;++k)result=result*x;return result;}
}
