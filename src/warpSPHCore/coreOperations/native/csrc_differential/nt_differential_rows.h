#pragma once
#include "nt_differential_backward.h"
#include "execution/neighborhoodAdjoint.h"
#include "execution/csrView.h"
namespace sph_native::differential {
using neighborhood::Owner;

struct Buffers {
    OptionalTensors directions, gradients;
    Tensors transpose;
    Tensor cotangent, output, cache;
    bool consistent=false;
};

template <class T,int D> struct Pullback {
    RowArgs<T,D> args;
    Arrays<T> arrays;
    const T *cotangent,*cache;
    int cache_width;
    bool consistent,geometry;
    template <Owner Side> SPH_HD int64_t prepare(int64_t owner) const { return owner; }
    template <Owner Side> SPH_HD auto identity() const {
        return neighborhood::Sums<T,Side==Owner::Query ? Layout<D>::Q : Layout<D>::Ref>{};
    }
    template <Owner Side,class State>
    SPH_HD void interact(int64_t owner,int64_t,neighborhood::Neighbor neighbor,State& sum) const {
        const int64_t i=Side==Owner::Query ? owner:neighbor.reference;
        const int64_t j=Side==Owner::Query ? neighbor.reference:owner;
        if (!query_allowed(args.qkind[i],args.direction) || !reference_allowed(args.rkind[j],args.direction)) return;
        const auto p=differential::prepare<T,D,T>(args,arrays,i,j,consistent);
        const T* incoming=cotangent+i*args.outputs;
        T final_bars[2]={};
        if (args.operation==Volume) {
            const T total=cache[i*cache_width];
            final_bars[0]=total==0 ? T(0):-incoming[0]/(total*total); incoming=final_bars;
        } else if (args.operation==Consistency) {
            const T numerator=cache[i*cache_width],denominator=cache[i*cache_width+1];
            if (denominator!=0) { final_bars[0]=incoming[0]/denominator; final_bars[1]=-incoming[0]*numerator/(denominator*denominator); }
            incoming=final_bars;
        }
        const auto bars=differential::backward(args,arrays,i,j,p,incoming,consistent,geometry);
        for (int c=0; c<(Side==Owner::Query ? Layout<D>::Q : Layout<D>::Ref); ++c) {
            if constexpr (Side==Owner::Query) sum.values[c]+=bars.query[c];
            else sum.values[c]+=bars.reference[c];
        }
    }
    template <Owner Side,class State> SPH_HD void finish(int64_t owner,const State& sum) const {
        for (int slot=0; slot<Slots; ++slot) {
            const int start=Layout<D>::offset(slot);
            if (!arrays.bars[slot] || start<0 || Layout<D>::reference(slot)!=(Side==Owner::Reference)) continue;
            const int width=Layout<D>::width(slot,args.fields);
            for (int c=0; c<width; ++c) arrays.bars[slot][owner*width+c]=sum.values[start+c];
        }
    }
};

template <class T,int D,int Mode> struct ForwardRow {
    RowArgs<T,D> args;
    Arrays<T> arrays;
    bool consistent;
    SPH_HD void operator()(int64_t i) const {
        using S=std::conditional_t<Mode==0,T,Directional<T>>;
        S sum[MaxOutputs]={};
        if (query_allowed(args.qkind[i],args.direction)) {
            const auto start=args.offsets[i];
            for (int32_t k=0; k<args.counts[i]; ++k) {
                const auto j=args.neighbors[start+k];
                if (!reference_allowed(args.rkind[j],args.direction)) continue;
                const auto pair=differential::prepare<T,D,S>(args,arrays,i,j,consistent);
                forward(args,pair,sum);
            }
        }
        for (int c=0; c<args.outputs; ++c) {
            if constexpr (Mode==0) args.output[i*args.outputs+c]=sum[c];
            else {
                T result=sum[c].tangent;
                if (args.operation==Volume) result=sum[0].value==0 ? T(0):-sum[0].tangent/(sum[0].value*sum[0].value);
                else if (args.operation==Consistency) result=sum[1].value==0 ? T(0):
                    (sum[0].tangent*sum[1].value-sum[0].value*sum[1].tangent)/(sum[1].value*sum[1].value);
                args.output[i*args.outputs+c]=result;
            }
        }
    }
};

// The same owned row runs on CPU and GPU. One thread owns one row here; the
// tuned field-only cooperative executor remains available independently.
template <int Mode> struct Rows {
    const OptionalTensors& flat;
    const Buffers& buffers;
    template <class T,int D,class Launch>
    void operator()(int64_t n,RowArgs<T,D> a,const Launch& launch) const {
        Arrays<T> arrays{};
        for (int slot=0; slot<Slots; ++slot) {
            arrays.values[slot]=flat[slot] && (flat[slot]->scalar_type()==ScalarType::Float || flat[slot]->scalar_type()==ScalarType::Double) ? ptr<T>(*flat[slot]):nullptr;
            if (!buffers.directions.empty() && buffers.directions[slot]) arrays.tangents[slot]=ptr<T>(*buffers.directions[slot]);
            if (!buffers.gradients.empty() && buffers.gradients[slot]) arrays.bars[slot]=buffers.gradients[slot]->mutable_data_ptr<T>();
        }
        if constexpr (Mode!=2) {
            a.output=buffers.output.mutable_data_ptr<T>();
            launch(n,ForwardRow<T,D,Mode>{a,arrays,buffers.consistent});
        } else {
            const bool geometry=arrays.bars[QX] || arrays.bars[RX] || arrays.bars[QH] || arrays.bars[RH] || arrays.bars[LO] || arrays.bars[HI];
            const Pullback<T,D> pullback{a,arrays,ptr<T>(buffers.cotangent),ptr<T>(buffers.cache),
                buffers.cache.defined() ? int(buffers.cache.size(1)):0,buffers.consistent,geometry};
            bool query=false,reference=false;
            for (int slot=0; slot<Slots; ++slot) if (arrays.bars[slot] && Layout<D>::offset(slot)>=0) {
                if (Layout<D>::reference(slot)) reference=true; else query=true;
            }

            // IndexView supports independent CSR ID/offset widths without
            // compiling a separate numerical policy for every combination.
            using Dynamic=neighborhood::CsrView;
            if (query) {
                const Dynamic csr{a.offsets,a.counts,a.neighbors};
                launch(n,neighborhood::OwnedAdjointRow<Owner::Query,decltype(pullback),Dynamic>{pullback,csr});
            }
            if (reference) {
                const auto& t=buffers.transpose;
                const Dynamic csc{index_view(t[0]),t[1].const_data_ptr<int32_t>(),index_view(t[2])};
                launch(a.references,neighborhood::OwnedAdjointRow<Owner::Reference,decltype(pullback),Dynamic>{pullback,csc});
            }
        }
    }
};
} // namespace sph_native::differential
