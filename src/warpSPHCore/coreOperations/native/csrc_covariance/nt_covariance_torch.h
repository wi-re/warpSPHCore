#pragma once
#include "torch/types.h"
namespace sph_native::neighborhood {
struct CovarianceCorrectionOptions {
    const Tensor& counts;
    template <class Operation> void operator()(Operation& op) const {
        op.fields.counts = counts.defined() ? counts.mutable_data_ptr<int32_t>() : nullptr;
    }
};
} // namespace sph_native::neighborhood
