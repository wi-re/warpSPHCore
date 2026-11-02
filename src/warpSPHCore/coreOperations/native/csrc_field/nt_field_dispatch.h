#pragma once
#include "nt_field_rows.h"
namespace sph_native {
template <class Executor, class Launch>
void execute_field(const Tensors& t, const OptionalTensors& corrections,
        const Executor& executor, int operation, int support, int direction, int kernel,
        int scheme, int lap, bool dot, bool consistent, const Launch& launch) {
    using namespace neighborhood;
    if (operation == 1)
        execute(CorrectedRows<CorrectedInterpolate, Executor>{corrections, {}, executor},
                t, support, direction, operation, launch, kernel, scheme, lap, dot);
    else if (operation == 2)
        execute(CorrectedRows<CorrectedGradient, Executor>{corrections, {}, executor},
                t, support, direction, operation, launch, kernel, scheme, lap, dot);
    else if (operation == 3)
        execute(CorrectedRows<CorrectedDivergence, Executor, DivergenceCorrectionOptions>{
                    corrections, {consistent}, executor},
                t, support, direction, operation, launch, kernel, scheme, lap, dot);
    else if (operation == 4)
        execute(CorrectedRows<CorrectedCurl, Executor>{corrections, {}, executor},
                t, support, direction, operation, launch, kernel, scheme, lap, dot);
    else
        execute(CorrectedRows<CorrectedLaplacian, Executor>{corrections, {}, executor},
                t, support, direction, operation, launch, kernel, scheme, lap, dot);
}
} // namespace sph_native
