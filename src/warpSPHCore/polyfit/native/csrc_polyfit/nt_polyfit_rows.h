#pragma once
#include "execution/dispatch.h"
#include "execution/numeric.h"
#include "execution/neighborhood.h"
#include "math/queryGeometry.h"
#include "kernels/kernel.h"
#include "util/directionality.h"

namespace sph_native::polyfit {
// One output entry per row keeps arbitrary polynomial order out of fixed-size
// register arrays. Derivative ownership is CSR for queries and CSC for sources.
struct Buffers {
    Tensor exponents, entries, volumes, values, output, counts, cotangent;
    Tensors tangents, bars;
    IndexView transposeOffsets, transposeQueries;
    const int32_t* transposeCounts=nullptr;
    int basis, components, sector, sectors;
    bool rhs, subtract, unit;
};
template<class T> SPH_HD T integer_power(T value,int exponent) {
    T result=1;
    for (int k=0;k<exponent;++k) result*=value;
    return result;
}
template<class T,int D> struct Contribution {
    T value=0, position[D]={}, support=0, mass=0, density=0, volume=0;
    T queryField=0, referenceField=0;
    int component=0;
};
template<class T,int D> struct Pair {
    RowArgs<T,D> args;
    const int32_t *exponents,*entries;
    const T *volumes,*values;
    int basis,components,sector,sectors;
    bool rhs,subtract,unit;
    SPH_HD int width() const { return rhs ? basis*components:basis*(basis+1)/2; }
    template<bool Derivatives> SPH_HD Contribution<T,D> evaluate(int64_t i,int64_t j,int channel) const {
        Contribution<T,D> result{};
        if (!query_allowed(args.qkind[i],args.direction) || !reference_allowed(args.rkind[j],args.direction)) return result;
        T dx[D];const T distance=QueryGeometry<T,D>(args.qx+i*D,args.lo,args.hi,args.periodic).distance(args.rx+j*D,dx);
        if (sector>=0 && i!=j) {
            bool allowed;
            if constexpr(D==1) allowed=sector==0 ? dx[0]>T(0):dx[0]<T(0);
            else {
                T angle=::atan2(-dx[1],-dx[0]);
                if (angle<T(0)) angle+=T(6.283185307179586);
                const T width=T(6.283185307179586)/T(sectors),low=T(sector)*width;
                allowed=angle>=low && angle<low+width;
            }
            if (!allowed) return result;
        }
        const T h=args.qh[i];
        if (h<=T(0)) return result;
        T kernel=0,radial=0;
        if (unit) kernel=distance<h ? T(1):T(0);
        else kernel_pair_value<T,D>(distance,h,args.kernel,kernel,radial);
        const T volume=unit ? T(1):volumes ? volumes[j]:args.mass[j]/args.density[j];
        const T weight=volume*kernel;
        const int a=rhs ? channel/components:entries[channel*2];
        const int b=rhs ? -1:entries[channel*2+1];
        T coordinate[D],factors[D],monomial=1;
        int powers[D];
        for (int d=0;d<D;++d) {
            coordinate[d]=-dx[d]/h;
            powers[d]=exponents[a*D+d]+(b>=0 ? exponents[b*D+d]:0);
            factors[d]=integer_power(coordinate[d],powers[d]);
            monomial*=factors[d];
        }
        result.component=rhs ? channel%components:0;
        const T field=rhs ? values[j*components+result.component]-(subtract ? values[i*components+result.component]:T(0)):T(1);
        result.value=weight*monomial*field;
        if constexpr(Derivatives) {
            for (int d=0;d<D;++d) {
                T derivative=powers[d] ? T(powers[d])*integer_power(coordinate[d],powers[d]-1):T(0);
                for (int c=0;c<D;++c) if (c!=d) derivative*=factors[c];
                result.position[d]=(volume*radial*dx[d]*monomial-weight*derivative/h)*field;
                result.support-=weight*derivative*coordinate[d]/h*field;
            }
            if (!unit) {
                result.support+=volume*(-T(D)*kernel-radial*distance*distance)/h*monomial*field;
                const T volumeBar=kernel*monomial*field;
                if (volumes) result.volume=volumeBar;
                else {
                    result.mass=volumeBar/args.density[j];
                    result.density=-volumeBar*args.mass[j]/(args.density[j]*args.density[j]);
                }
            }
            if (rhs) {
                result.referenceField=weight*monomial;
                result.queryField=subtract ? -result.referenceField:T(0);
            }
        }
        return result;
    }
};
template<class T,int D> struct Rows {
    Pair<T,D> pair;
    T* output;
    int32_t* counts;
    const T* cotangent;
    const T* tangents[8]={};
    T* bars[8]={};
    IndexView transposeOffsets,transposeQueries;
    const int32_t* transposeCounts;
    int role;
    // Geometry slots: query x, reference x, query h, source mass/density,
    // explicit source volumes, shared field values, domain lengths.
    SPH_HD T tangent(int slot,int64_t index) const {return tangents[slot] ? tangents[slot][index]:T(0);}
    SPH_HD T winding(int64_t i,int64_t j,int d,T dx) const {
        return pair.args.periodic[d] ? ::round((pair.args.qx[i*D+d]-pair.args.rx[j*D+d]-dx)/(pair.args.hi[d]-pair.args.lo[d])):T(0);
    }
    SPH_HD void operator()(int64_t thread) const {
        const auto& a=pair.args;
        const int width=pair.width();
        if (role<=1) {
            const int64_t i=thread/width;const int channel=int(thread%width);
            T sum=0;int32_t count=0;
            for (int32_t local=0;local<a.counts[i];++local) {
                const int64_t j=a.neighbors[a.offsets[i]+local];
                if (role==0) {
                    const auto contribution=pair.template evaluate<false>(i,j,channel);
                    sum+=contribution.value;
                    // Count nonzero stencil weights, independently of monomial
                    // zeros: the constant channel supplies that count below.
                    if (!pair.rhs && channel==0) {
                        const auto saved=pair;auto constant=saved;
                        // The dedicated count pass reuses the same stencil with
                        // an empty exponent, including LABFM's nonconstant basis.
                        int32_t zero[D]={};int32_t entry[2]={};
                        constant.exponents=zero;constant.entries=entry;
                        if (constant.template evaluate<false>(i,j,0).value!=T(0)) ++count;
                    }
                } else {
                    const auto c=pair.template evaluate<true>(i,j,channel);
                    T dx[D];a.displacement(i,j,dx);
                    T value=c.support*tangent(2,i)+c.mass*tangent(3,j)+c.density*tangent(4,j)+c.volume*tangent(5,j);
                    for (int d=0;d<D;++d) value+=c.position[d]*(tangent(0,i*D+d)-tangent(1,j*D+d)-winding(i,j,d,dx[d])*tangent(7,d));
                    if (pair.rhs) value+=c.referenceField*tangent(6,j*pair.components+c.component)+c.queryField*tangent(6,i*pair.components+c.component);
                    sum+=value;
                }
            }
            output[thread]=sum;
            if (role==0 && !pair.rhs && channel==0) counts[i]=count;
        } else if (role==2) {
            // Query-owned reverse reduction; per-query domain contributions
            // are reduced by Torch after this kernel, without floating atomics.
            const int64_t i=thread;
            for (int32_t local=0;local<a.counts[i];++local) {
                const int64_t j=a.neighbors[a.offsets[i]+local];
                T dx[D];a.displacement(i,j,dx);
                for (int channel=0;channel<width;++channel) {
                    const auto c=pair.template evaluate<true>(i,j,channel);const T bar=cotangent[i*width+channel];
                    for (int d=0;d<D;++d) {
                        bars[0][i*D+d]+=bar*c.position[d];
                        bars[7][i*D+d]-=bar*c.position[d]*winding(i,j,d,dx[d]);
                    }
                    bars[2][i]+=bar*c.support;
                    if (pair.rhs && pair.subtract) bars[6][i*pair.components+c.component]+=bar*c.queryField;
                }
            }
        } else {
            const int64_t j=thread;
            for (int32_t local=0;local<transposeCounts[j];++local) {
                const int64_t i=transposeQueries[transposeOffsets[j]+local];
                for (int channel=0;channel<width;++channel) {
                    const auto c=pair.template evaluate<true>(i,j,channel);const T bar=cotangent[i*width+channel];
                    for (int d=0;d<D;++d) bars[1][j*D+d]-=bar*c.position[d];
                    bars[3][j]+=bar*c.mass;bars[4][j]+=bar*c.density;
                    if (bars[5]) bars[5][j]+=bar*c.volume;
                    if (pair.rhs) bars[6][j*pair.components+c.component]+=bar*c.referenceField;
                }
            }
        }
    }
};
template<class Launch> struct Dispatch {
    const Buffers& buffers;int role;
    template<class T,int D> void operator()(int64_t n,RowArgs<T,D> a,const Launch& launch) const {
        Rows<T,D> rows{};
        rows.pair={a,buffers.exponents.const_data_ptr<int32_t>(),buffers.entries.const_data_ptr<int32_t>(),
            ptr<T>(buffers.volumes),ptr<T>(buffers.values),buffers.basis,buffers.components,
            buffers.sector,buffers.sectors,buffers.rhs,buffers.subtract,buffers.unit};
        rows.output=ptr<T>(buffers.output) ? buffers.output.mutable_data_ptr<T>():nullptr;
        rows.counts=buffers.counts.defined() ? buffers.counts.mutable_data_ptr<int32_t>():nullptr;
        rows.cotangent=ptr<T>(buffers.cotangent);rows.role=role;
        rows.transposeOffsets=buffers.transposeOffsets;rows.transposeQueries=buffers.transposeQueries;rows.transposeCounts=buffers.transposeCounts;
        for (int slot=0;slot<8;++slot) {
            if (!buffers.tangents.empty()) rows.tangents[slot]=ptr<T>(buffers.tangents[slot]);
            if (!buffers.bars.empty() && buffers.bars[slot].defined() && buffers.bars[slot].numel()) rows.bars[slot]=buffers.bars[slot].mutable_data_ptr<T>();
        }
        launch(role<=1 ? n*rows.pair.width():role==2 ? n:a.references,rows);
    }
};
void run_gpu(const Tensors&,const Buffers&,int,int,int);
} // namespace sph_native::polyfit
