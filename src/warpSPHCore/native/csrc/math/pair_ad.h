#pragma once
#include "config.h"
#include "util/support.h"
#include <cassert>
#include <cstdint>
#include <cmath>

namespace sph_native::pair_ad {
// The trace is local to one pair, independent of neighborhood/population size.
// D<=3, seven limiter branches and both work modes fit this conservative bound.
// An assertion catches future algebra changes instead of truncating derivatives.
constexpr int DefaultNodes=1536,MaxLeaves=192;
template<class T,int Capacity=DefaultNodes>struct Trace {
    struct Node {T bar,da,db;int a,b;};
    struct Leaf {int slot,node;int64_t at;};
    Node nodes[Capacity];Leaf leaves[MaxLeaves];int count=0,inputs=0;
    SPH_HD int record(int a,int b,T da,T db) {
        assert(count<Capacity);const int n=count++;
        nodes[n]={T(0),da,db,a,b};return n;
    }
    SPH_HD int input(int slot,int64_t at) {
        assert(inputs<MaxLeaves);const int n=record(-1,-1,T(0),T(0));
        leaves[inputs++]={slot,n,at};return n;
    }
    SPH_HD void reverse() {
        for(int n=count-1;n>=0;--n) {
            const auto x=nodes[n];
            if(x.a>=0)nodes[x.a].bar+=x.da*x.bar;
            if(x.b>=0)nodes[x.b].bar+=x.db*x.bar;
        }
    }
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    // CPU owners may seed external, wider bars without changing scalar
    // values, local Jacobians, the default trace layout or its replay.
    template<class Accumulator>void reverse(Accumulator* bars)const {
        for(int n=count-1;n>=0;--n) {
            const auto x=nodes[n];
            if(x.a>=0)bars[x.a]+=Accumulator(x.da)*bars[n];
            if(x.b>=0)bars[x.b]+=Accumulator(x.db)*bars[n];
        }
    }
#endif
};
template<class T,bool Reverse,int Capacity=DefaultNodes>struct Number {
    T v=0,d=0;int node=-1;Trace<T,Capacity>* trace=nullptr;
    SPH_HD Number()=default;
    SPH_HD Number(T value):v(value){}
    SPH_HD static Number leaf(T value,T tangent,Trace<T,Capacity>* t,int slot,int64_t at,bool active) {
        Number a(value);if(active){a.d=tangent;a.trace=t;if constexpr(Reverse)a.node=t->input(slot,at);else a.node=0;}return a;
    }
    SPH_HD static Number binary(T value,Number a,Number b,T da,T db) {
        Number c(value);
        if(a.node<0&&b.node<0)return c;
        if constexpr(Reverse) {
            c.trace=a.trace?a.trace:b.trace;c.node=c.trace->record(a.node,b.node,da,db);
        } else {c.node=0;if(a.node>=0)c.d+=da*a.d;if(b.node>=0)c.d+=db*b.d;}
        return c;
    }
    SPH_HD static Number unary(T value,Number a,T da){return binary(value,a,Number(0),da,T(0));}
    SPH_HD Number& operator+=(Number b){return *this=*this+b;}
    SPH_HD Number& operator-=(Number b){return *this=*this-b;}
    SPH_HD Number& operator*=(Number b){return *this=*this*b;}
    SPH_HD void seed(T b)const{if constexpr(Reverse){if(node>=0)trace->nodes[node].bar+=b;}}
};
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> operator+(Number<T,R,Capacity>a,Number<T,R,Capacity>b){return Number<T,R,Capacity>::binary(a.v+b.v,a,b,T(1),T(1));}
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> operator-(Number<T,R,Capacity>a,Number<T,R,Capacity>b){return Number<T,R,Capacity>::binary(a.v-b.v,a,b,T(1),T(-1));}
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> operator-(Number<T,R,Capacity>a){return Number<T,R,Capacity>::unary(-a.v,a,T(-1));}
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> operator*(Number<T,R,Capacity>a,Number<T,R,Capacity>b){return Number<T,R,Capacity>::binary(a.v*b.v,a,b,b.v,a.v);}
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> operator/(Number<T,R,Capacity>a,Number<T,R,Capacity>b){return Number<T,R,Capacity>::binary(a.v/b.v,a,b,T(1)/b.v,-a.v/(b.v*b.v));}
#define CRK_COMPARE(OP) template<class T,bool R,int Capacity>SPH_HD bool operator OP(Number<T,R,Capacity>a,Number<T,R,Capacity>b){return a.v OP b.v;}
CRK_COMPARE(<) CRK_COMPARE(>) CRK_COMPARE(<=) CRK_COMPARE(>=) CRK_COMPARE(==) CRK_COMPARE(!=)
#undef CRK_COMPARE
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> abs(Number<T,R,Capacity>a){return Number<T,R,Capacity>::unary(std::abs(a.v),a,a.v<T(0)?T(-1):T(1));}
// CRK uses safe_sqrt, whose adjoint stops at a zero argument.
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> sqrt(Number<T,R,Capacity>a){const T v=std::sqrt(a.v);return a.v>T(0)?Number<T,R,Capacity>::unary(v,a,T(1)/(T(2)*v)):Number<T,R,Capacity>(v);}
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> exp(Number<T,R,Capacity>a){const T v=std::exp(a.v);return Number<T,R,Capacity>::unary(v,a,v);}
template<class T,bool R,int Capacity>SPH_HD Number<T,R,Capacity> pow(Number<T,R,Capacity>a,Number<T,R,Capacity>b){const T v=std::pow(a.v,b.v);return Number<T,R,Capacity>::binary(v,a,b,b.v*std::pow(a.v,b.v-T(1)),v*std::log(a.v));}
template<class T,bool R,int Capacity>SPH_HD bool isfinite(Number<T,R,Capacity>a){return std::isfinite(a.v);}
template<class T,bool R,int Capacity>SPH_HD bool isnan(Number<T,R,Capacity>a){return std::isnan(a.v);}
// Attach a custom primitive Jacobian without differentiating its regularized
// primal implementation. Used for Warp's kernel-gradient Hessian contract.
template<class T,bool R,int Capacity,int N>SPH_HD Number<T,R,Capacity> primitive(T value,const Number<T,R,Capacity>(&x)[N],const T(&partial)[N]) {
    Number<T,R,Capacity> result(value);
    for(int k=0;k<N;++k)if(x[k].node>=0)
        result=Number<T,R,Capacity>::binary(value,result,x[k],T(1),partial[k]);
    return result;
}
// Keep the selected dependency separate from the exact floating-point value:
// signed-zero ties can select different operands in forward and adj_max.
template<class T,bool R,int N>SPH_HD Number<T,R,N> maximum_support(Number<T,R,N> a,Number<T,R,N> b) {
    auto result=maximum_support_query(a.v,b.v)?a:b;
    result.v=sph_native::maximum_support(a.v,b.v);return result;
}
}
