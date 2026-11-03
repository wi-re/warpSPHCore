#pragma once
#include <limits>
#include <algorithm>
#include "execution/dispatch.h"

namespace sph_native::verlet {
constexpr int BlockSize = 256;

// The same particle calculation and associative reduction are used on CPU,
// CUDA and HIP. These are the motion/support budgets needed by Verlet reuse,
// not a verification pass over CSR storage.
template <class T> struct Metric {
    T distance = 0, drift = 0, minimum = std::numeric_limits<T>::infinity();
    SPH_HD void merge(const Metric& other) {
        distance = distance > other.distance ? distance : other.distance;
        drift = drift > other.drift ? drift : other.drift;
        minimum = minimum < other.minimum ? minimum : other.minimum;
    }
    SPH_HD void store(T* output) const {
        output[0] = distance; output[1] = drift; output[2] = minimum;
    }
    SPH_HD static Metric load(const T* input) { return {input[0], input[1], input[2]}; }
};

template <class T, int D> struct Population {
    int64_t size;
    const T *positions, *prior_positions, *supports, *prior_supports;
    const T *lo, *hi, *scale;
    const bool* periodic;
    SPH_HD Metric<T> operator()(int64_t i) const {
        T displacement[D];
        const T distance = minimum_image_displacement<T,D>(positions+i*D, prior_positions+i*D,
                                                           lo, hi, periodic, displacement);
        const T before = prior_supports[i] / *scale, now = supports[i];
        const T delta = before-now;
        return {distance, delta < 0 ? -delta : delta, before < now ? before : now};
    }
};

template <class T> SPH_HD void finish(Metric<T> query, Metric<T> reference, T scale,
                                      int support_case, T slack, T* output, bool* flag) {
    const T motion = query.distance + reference.distance;
    Metric<T> support = query;
    if (support_case == 1) support = reference;
    else if (support_case == 2) support.merge(reference);
    const T buffer = (scale-T(1))*support.minimum;
    output[0] = motion; output[1] = support.drift;
    output[2] = support.minimum; output[3] = buffer;
    *flag = motion + support.drift > buffer*(T(1)-slack);
}

struct Candidates { IndexView neighbors, offsets; const int32_t* counts; };
template <class T, int D> struct FilterRow {
    RowArgs<T,D> args;
    Candidates source;
    bool collect;
    SPH_HD void operator()(int64_t i) const {
        int32_t found = 0;
        for (int32_t k=0; k<source.counts[i]; ++k) {
            const int64_t j = source.neighbors[source.offsets[i]+k];
            T delta[D];
            // Match Warp's filtering rule, including the exact cutoff. Fresh
            // radius search has its separate, strict '<' membership rule.
            if (args.displacement(i,j,delta) <= args.radius(i,j)) {
                if (collect) args.collected[args.offsets[i]+found] = j;
                ++found;
            }
        }
        if (!collect) args.counts[i] = found;
    }
};
struct FilterRows {
    Candidates source;
    bool collect;
    template <class T, int D, class Launcher>
    void operator()(int64_t size, RowArgs<T,D> args, const Launcher& launch) const {
        launch(size, FilterRow<T,D>{args,source,collect});
    }
};

struct ValidityInputs {
    Tensor prior_query, prior_reference, prior_query_support, prior_reference_support, scale;
};
inline int64_t blocks(int64_t size) { return std::min<int64_t>((size+BlockSize-1)/BlockSize,65535); }
template <class T, int D>
Population<T,D> population(const RowArgs<T,D>& args, const ValidityInputs& inputs,
                           int64_t size, bool reference) {
    return {size, reference ? args.rx : args.qx,
            ptr<T>(reference ? inputs.prior_reference : inputs.prior_query),
            reference ? args.rh : args.qh,
            ptr<T>(reference ? inputs.prior_reference_support : inputs.prior_query_support),
            args.lo,args.hi,ptr<T>(inputs.scale),args.periodic};
}
void validity_gpu(const Tensors&, const ValidityInputs&, int, double, bool,
                  const Tensor&, const Tensor&, const Tensor&);
void filter_gpu(const Tensors&, int, Candidates, bool);
} // namespace sph_native::verlet
