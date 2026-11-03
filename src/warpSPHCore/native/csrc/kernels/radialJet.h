#pragma once
#include "config.h"
namespace sph_native {
// Radial value and its first three analytic derivatives. Product rules keep
// factored polynomial definitions together; no Torch/Warp AD is involved.
template <class T> struct RadialJet {
    T v, d1, d2, d3;
    SPH_HD RadialJet(T value = T(0), T a = T(0), T b = T(0), T c = T(0))
        : v(value), d1(a), d2(b), d3(c) {}
    SPH_HD RadialJet operator+(RadialJet b) const {
        return {v + b.v, d1 + b.d1, d2 + b.d2, d3 + b.d3};
    }
    SPH_HD RadialJet operator-(RadialJet b) const {
        return {v - b.v, d1 - b.d1, d2 - b.d2, d3 - b.d3};
    }
    SPH_HD RadialJet operator*(RadialJet b) const {
        return {v * b.v, d1 * b.v + v * b.d1, d2 * b.v + T(2) * d1 * b.d1 + v * b.d2,
                d3 * b.v + T(3) * d2 * b.d1 + T(3) * d1 * b.d2 + v * b.d3};
    }
    SPH_HD RadialJet operator*(T b) const { return {v * b, d1 * b, d2 * b, d3 * b}; }
};
template <class T> SPH_HD RadialJet<T> jet_power(RadialJet<T> x, int n) {
    RadialJet<T> result(T(1));
    for (int i = 0; i < n; ++i)
        result = result * x;
    return result;
}
template <class T> SPH_HD RadialJet<T> positive_power(T knot, RadialJet<T> q, int n) {
    // At an internal cubic knot the right branch owns the third derivative.
    if (q.v >= knot && !(knot == T(1) && q.v == T(1)))
        return {};
    return jet_power(RadialJet<T>(knot) - q, n);
}
} // namespace sph_native
