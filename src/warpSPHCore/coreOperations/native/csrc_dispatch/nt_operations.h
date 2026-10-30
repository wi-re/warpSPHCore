#pragma once
#include "coreOperations/native/csrc_density/nt_density_forward.h"
#include "coreOperations/native/csrc_density/nt_density_adjoint.h"
#include "coreOperations/native/csrc_interpolate/nt_interpolate_forward.h"
#include "coreOperations/native/csrc_interpolate/nt_interpolate_adjoint.h"
#include "coreOperations/native/csrc_gradient/nt_gradient_forward.h"
#include "coreOperations/native/csrc_gradient/nt_gradient_adjoint.h"
#include "coreOperations/native/csrc_divergence/nt_divergence_forward.h"
#include "coreOperations/native/csrc_divergence/nt_divergence_adjoint.h"
#include "coreOperations/native/csrc_curl/nt_curl_forward.h"
#include "coreOperations/native/csrc_curl/nt_curl_adjoint.h"
#include "coreOperations/native/csrc_laplacian/nt_laplacian_forward.h"
#include "coreOperations/native/csrc_laplacian/nt_laplacian_adjoint.h"
#include "coreOperations/native/csrc_covariance/nt_covariance_forward.h"
#include "coreOperations/native/csrc_covariance/nt_covariance_adjoint.h"

#include "runtime/backend.h"
namespace sph_native::runtime {
template <class T, int D, class Launch>
void run_rows(Task task, const Prepared &p, RowArgs<T, D> a, Buffer out, ConstBuffer cot,
              const Adjoints &gradients, const Launch &launch) {
    GradientBuffers<T> bars{};
    bars.mask = gradients.requested & used_gradients(p.properties);
    for (int s = 0; s < OUTPUT; ++s)
        bars.pointers[s] = static_cast<T *>(gradients.slots[s].data);
    switch (p.properties.operation) {
    case Operation::Density:
        if (task == Task::Forward)
            launch(p.rows, DensityRow<T, D>{a});
        else
            launch(p.rows, AdjointRow<T, D, DensityAdjointPair<T, D>>{
                               a, static_cast<const T *>(cot.data), bars});
        break;
    case Operation::Interpolate:
        if (task == Task::Forward)
            launch(p.rows, InterpolateRow<T, D>{a});
        else
            launch(p.rows, AdjointRow<T, D, InterpolateAdjointPair<T, D>>{
                               a, static_cast<const T *>(cot.data), bars});
        break;
    case Operation::Gradient:
        if (task == Task::Forward)
            launch(p.rows, GradientRow<T, D>{a});
        else
            launch(p.rows, AdjointRow<T, D, GradientAdjointPair<T, D>>{
                               a, static_cast<const T *>(cot.data), bars});
        break;
    case Operation::Divergence:
        if (task == Task::Forward)
            launch(p.rows, DivergenceRow<T, D>{a});
        else
            launch(p.rows, AdjointRow<T, D, DivergenceAdjointPair<T, D>>{
                               a, static_cast<const T *>(cot.data), bars});
        break;
    case Operation::Curl:
        if (task == Task::Forward)
            launch(p.rows, CurlRow<T, D>{a});
        else
            launch(p.rows, AdjointRow<T, D, CurlAdjointPair<T, D>>{
                               a, static_cast<const T *>(cot.data), bars});
        break;
    case Operation::Laplacian:
        if (task == Task::Forward)
            launch(p.rows, LaplacianRow<T, D>{a});
        else
            launch(p.rows, AdjointRow<T, D, LaplacianAdjointPair<T, D>>{
                               a, static_cast<const T *>(cot.data), bars});
        break;
    case Operation::Covariance:
        if (task == Task::Forward)
            launch(p.rows, CovarianceRow<T, D>{a});
        else
            launch(p.rows, AdjointRow<T, D, CovarianceAdjointPair<T, D>>{
                               a, static_cast<const T *>(cot.data), bars});
        break;
    }
}
} // namespace sph_native::runtime
