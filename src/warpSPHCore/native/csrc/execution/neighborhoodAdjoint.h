#pragma once
#include "execution/neighborhood.h"

namespace sph_native::neighborhood {

enum class Owner { Query, Reference };

// Query ownership traverses CSR; reference ownership traverses its transpose.
// Pullback always differentiates the original directed interaction. A flipped
// neighborhood is an execution choice, never a swap of the physics arguments.
template <Owner Side, class Pullback, class Prepared, class State> struct PullbackVisit {
    const Pullback& pullback;
    const Prepared& prepared;
    int64_t owner;
    State& state;

    SPH_HD void operator()(Neighbor neighbor) const {
        pullback.template interact<Side>(owner, prepared, neighbor, state);
    }
};

template <Owner Side, class Pullback, class Traversal> struct OwnedAdjointRow {
    Pullback pullback;
    Traversal traversal;

    template <class Reduction>
    SPH_HD void run(int64_t owner, int lane, int lanes, const Reduction& reduction) const {
        const auto prepared = pullback.template prepare<Side>(owner);
        auto state = pullback.template identity<Side>();
        traversal.each(owner, lane, lanes,
            PullbackVisit<Side, Pullback, decltype(prepared), decltype(state)>{
                pullback, prepared, owner, state});
        state.reduce(reduction);
        if (lane == 0) pullback.template finish<Side>(owner, state);
    }

    SPH_HD void operator()(int64_t owner) const {
        run(owner, 0, 1, LocalReduction{});
    }
};

} // namespace sph_native::neighborhood
