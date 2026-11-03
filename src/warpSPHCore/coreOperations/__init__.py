from .._backend import BACKEND

if BACKEND == 'native':
    from .native import computeDensity, computeGradient, sphOperation
    from .native.nt_density_hvp import computeSPHDensityPositionHVP
    from .native.nt_jvp import warpOperationJVP
    from .native.nt_geometry_jvp import *
    from .native.nt_geometry_jvp import __all__ as _products
    _computeSPHDensity_stateBackend = computeDensity
    _computeSPHGradient_stateBackend = computeGradient
    __all__ = ['computeDensity', 'computeGradient', 'sphOperation', 'computeSPHDensityPositionHVP', 'warpOperationJVP',
               '_computeSPHDensity_stateBackend', '_computeSPHGradient_stateBackend'] + _products
else:
    from .wp_covariance import _computeSPHCovariance_stateBackend
    from .wp_covarianceJVP import computeCovarianceGeometryJVP
    from .wp_curl import _computeSPHCurl_stateBackend
    from .native import computeDensity as _computeSPHDensity_stateBackend
    from .wp_interpolate import _computeSPHInterpolant_stateBackend
    from .wp_laplacian import _computeSPHLaplacian_stateBackend
    from .wp_divergence import _computeSPHDivergence_stateBackend
    from .native import computeGradient as _computeSPHGradient_stateBackend
    from .wp_densityJVP import computeSPHDensityGeometryJVP
    from .wp_densityHVP import computeSPHDensityPositionHVP
    from .wp_interpolateJVP import computeSPHInterpolateGeometryJVP
    from .wp_gradientJVP import computeSPHGradientGeometryJVP
    from .wp_divergenceJVP import computeSPHDivergenceGeometryJVP
    from .wp_curlJVP import computeSPHCurlGeometryJVP
    from .wp_laplacianJVP import (
        computeSPHLaplacianGeometryJVP,
        computeSPHLaplacianBrookshawGeometryJVP,
        computeSPHLaplacianNaiveGeometryJVP,
        computeSPHLaplacianDotGeometryJVP,
        computeSPHLaplacianDefaultGeometryJVP,
    )

    __all__ = [
        "_computeSPHCovariance_stateBackend",
        "computeCovarianceGeometryJVP",
        "_computeSPHCurl_stateBackend",
        "_computeSPHDensity_stateBackend",
        "_computeSPHInterpolant_stateBackend",
        "_computeSPHLaplacian_stateBackend",
        "_computeSPHDivergence_stateBackend",
        "_computeSPHGradient_stateBackend",
        "computeSPHDensityGeometryJVP",
        "computeSPHDensityPositionHVP",
        "computeSPHInterpolateGeometryJVP",
        "computeSPHGradientGeometryJVP",
        "computeSPHDivergenceGeometryJVP",
        "computeSPHCurlGeometryJVP",
        "computeSPHLaplacianGeometryJVP",
        "computeSPHLaplacianBrookshawGeometryJVP",
        "computeSPHLaplacianNaiveGeometryJVP",
        "computeSPHLaplacianDotGeometryJVP",
        "computeSPHLaplacianDefaultGeometryJVP",
    ]
