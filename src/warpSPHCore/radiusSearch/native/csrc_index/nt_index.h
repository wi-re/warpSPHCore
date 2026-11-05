#pragma once
#include "radiusSearch/native/csrc_radius/nt_radius_dispatch.h"
#include "radiusSearch/native/csrc_compact/nt_compact_backend.h"
namespace sph_native::search_index {
enum Metadata { Version,Dimension,PrecisionBits,SupportMode,References,Occupied,Buckets,CellWidth,
                Lower=8,Upper=11,Cells=14,Periodic=17,RadiusBound=20,MetadataSize=21 };
// The retained index is ordinary Torch storage, not a custom class or global
// handle table. Fixed host metadata describes its three device-owned arrays.
template<class Index> struct FillRows {
    IndexView offsets;const int32_t* counts;Index* output;int64_t start;
    SPH_HD void operator()(int64_t i) const {
        for(int32_t k=0;k<counts[i];++k)output[offsets[i]+k]=Index(start+i);
    }
};
template<class Launch> void fill_rows(const Tensor& offsets,const Tensor& counts,const Tensor& output,int64_t start,const Launch& launch) {
    if(output.scalar_type()==ScalarType::Long)launch(counts.numel(),FillRows<int64_t>{index_view(offsets),counts.const_data_ptr<int32_t>(),output.mutable_data_ptr<int64_t>(),start});
    else launch(counts.numel(),FillRows<int32_t>{index_view(offsets),counts.const_data_ptr<int32_t>(),output.mutable_data_ptr<int32_t>(),start});
}
void copy_gpu(const Tensor&,const void*,size_t);
void fill_gpu(const Tensor&,const Tensor&,const Tensor&,int64_t);
}

namespace sph_native::search_index {
struct FixedKeys {
    const int32_t* cells;const int32_t* counts;int64_t* output;int dim;uint32_t buckets;bool hash;
    SPH_HD void operator()(int64_t i) const {
        if(hash) {
            constexpr uint32_t primes[3]={73856093u,19349663u,83492791u};
            uint32_t value=0;
            for(int d=0;d<dim;++d)value+=uint32_t(cells[i*int64_t(dim)+d])*(dim==1 ? 1u:primes[d]);
            output[i]=int64_t(value%buckets);
        } else {
            int64_t value=0,stride=1;
            for(int d=0;d<dim;++d){value+=int64_t(cells[i*int64_t(dim)+d])*stride;stride*=int64_t(counts[d]);}
            output[i]=value;
        }
    }
};
template<class Launch> void fixed_keys(const Tensor& cells,const Tensor& counts,const Tensor& output,int64_t buckets,const Launch& launch) {
    launch(cells.size(0),FixedKeys{cells.const_data_ptr<int32_t>(),counts.const_data_ptr<int32_t>(),output.mutable_data_ptr<int64_t>(),int(cells.size(1)),uint32_t(buckets),buckets!=0});
}
void fixed_keys_gpu(const Tensor&,const Tensor&,const Tensor&,int64_t);
}
