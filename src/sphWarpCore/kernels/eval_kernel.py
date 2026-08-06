
from .kernelFunctions import *
from ..types import *
from typing import Any
import warp as wp
import numpy as np
from ..math import computeDistanceVec
from ..enumTypes import KernelFunctions, SupportScheme

@wp.func
def eval_k(q: scalar_t, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_k(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_k(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_k(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_k(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_k(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_k(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_k(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_k(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):
        return cohesionKernel_k(q, dim)
    return scalar_t(np.nan) 

@wp.func
def eval_dkdq(q: scalar_t, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_dkdq(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):
        return cohesionKernel_dkdq(q, dim)
    return scalar_t(np.nan)

@wp.func
def eval_d2kdq2(q: scalar_t, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_d2kdq2(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_d2kdq2(q, dim)
    return scalar_t(np.nan)

@wp.func
def eval_d3kdq3(q: scalar_t, dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_d3kdq3(q, dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_d3kdq3(q, dim)
    return scalar_t(np.nan)

@wp.func
def eval_C_d(dim: wp.int32, kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_C_d(dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_C_d(dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_C_d(dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_C_d(dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_C_d(dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_C_d(dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_C_d(dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_C_d(dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_C_d(dim)
    return scalar_t(np.nan)

@wp.func
def eval_kernelScale(kernel: wp.int32, dim: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_kernelScale(dim)
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_kernelScale(dim)
    return scalar_t(np.nan)

@wp.func
def eval_packing(kernel: wp.int32):
    if kernel == wp.static(KernelFunctions.Wendland2.value):
        return wendland2_packingRatio()
    elif kernel == wp.static(KernelFunctions.Wendland4.value):
        return wendland4_packingRatio()
    elif kernel == wp.static(KernelFunctions.Wendland6.value):
        return wendland6_packingRatio()
    elif kernel == wp.static(KernelFunctions.CubicSpline.value):
        return cubicSpline_packingRatio()
    elif kernel == wp.static(KernelFunctions.QuarticSpline.value):
        return quarticSpline_packingRatio()
    elif kernel == wp.static(KernelFunctions.QuinticSpline.value):
        return quinticSpline_packingRatio()
    elif kernel == wp.static(KernelFunctions.B7.value):
        return B7_packingRatio()
    elif kernel == wp.static(KernelFunctions.Poly6.value):
        return poly6_packingRatio()
    elif kernel == wp.static(KernelFunctions.Spiky.value):
        return spiky_packingRatio()
    elif kernel == wp.static(KernelFunctions.ViscosityKernel.value):
        return viscosityKernel_packingRatio()
    elif kernel == wp.static(KernelFunctions.AdhesionKernel.value):
        return adhesionKernel_packingRatio()
    elif kernel == wp.static(KernelFunctions.CohesionKernel.value):         
        return cohesionKernel_packingRatio()
    return scalar_t(np.nan)
