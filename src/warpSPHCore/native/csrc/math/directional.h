#pragma once
#include "config.h"
namespace sph_native {
// A first-order scalar direction used only inside a single particle pair.
// There is no tape, heap allocation, or cross-operation differentiation graph.
template <class T> struct Directional {
    T value, tangent;
    SPH_HD Directional(T value = T(0), T tangent = T(0)) : value(value), tangent(tangent) {}
    SPH_HD Directional operator-() const { return {-value, -tangent}; }
    SPH_HD Directional& operator+=(Directional r) { value += r.value; tangent += r.tangent; return *this; }
    SPH_HD Directional& operator-=(Directional r) { value -= r.value; tangent -= r.tangent; return *this; }
};
template <class T> SPH_HD Directional<T> operator+(Directional<T> a, Directional<T> b) { return {a.value+b.value,a.tangent+b.tangent}; }
template <class T> SPH_HD Directional<T> operator-(Directional<T> a, Directional<T> b) { return {a.value-b.value,a.tangent-b.tangent}; }
template <class T> SPH_HD Directional<T> operator*(Directional<T> a, Directional<T> b) { return {a.value*b.value,a.tangent*b.value+a.value*b.tangent}; }
template <class T> SPH_HD Directional<T> operator/(Directional<T> a, Directional<T> b) { return {a.value/b.value,(a.tangent*b.value-a.value*b.tangent)/(b.value*b.value)}; }
template <class S> struct ScalarValue { using Type = S; SPH_HD static S get(S a) { return a; } };
template <class T> struct ScalarValue<Directional<T>> { using Type = T; SPH_HD static T get(Directional<T> a) { return a.value; } };
} // namespace sph_native
