"""Compatibility geometry-product names over one native first-order dispatcher."""
from ...enumTypes import WarpOperation,GradientScheme,LaplacianScheme,OperationDirection
from ..._models import OperationProperties
from .nt_jvp import warpOperationJVP


def _launch(operation,query,domain,kernel,support,adjacency,query_tangent,
            *,gradientMode=GradientScheme.Symmetric,laplacianMode=LaplacianScheme.Brookshaw,
            operationMode=OperationDirection.AllToAll,**options):
    properties=OperationProperties(kernel,operation=operation,supportMode=support,
        gradientMode=gradientMode,laplacianMode=laplacianMode,operationMode=operationMode)
    return warpOperationJVP(query,properties,domain,adjacency=adjacency,
                           queryTangentState=query_tangent,**options)


def computeCovarianceGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, referenceVolumes=None, tangentReferenceVolumes=None,
        operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Covariance,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        referenceVolumes=referenceVolumes, tangentReferenceVolumes=tangentReferenceVolumes,
        operationMode=operationMode)


def computeSPHCurlGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentQueryValues=None,
        tangentReferenceValues=None, referenceVolumes=None, tangentReferenceVolumes=None, crkState=None,
        crkTangentState=None, renormalizationState=None, renormalizationTangentState=None,
        gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Curl,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues, tangentQueryValues=tangentQueryValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, crkState=crkState, crkTangentState=crkTangentState,
        renormalizationState=renormalizationState, renormalizationTangentState=renormalizationTangentState,
        gradientMode=gradientMode, operationMode=operationMode)


def computeSPHDensityGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Density,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        operationMode=operationMode)


def computeSPHDivergenceGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentQueryValues=None,
        tangentReferenceValues=None, referenceVolumes=None, tangentReferenceVolumes=None, crkState=None,
        crkTangentState=None, renormalizationState=None, renormalizationTangentState=None,
        gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Divergence,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues, tangentQueryValues=tangentQueryValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, crkState=crkState, crkTangentState=crkTangentState,
        renormalizationState=renormalizationState, renormalizationTangentState=renormalizationTangentState,
        gradientMode=gradientMode, operationMode=operationMode)


def computeSPHGradientGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentQueryValues=None,
        tangentReferenceValues=None, referenceVolumes=None, tangentReferenceVolumes=None, crkState=None,
        crkTangentState=None, renormalizationState=None, renormalizationTangentState=None,
        gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Gradient,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues, tangentQueryValues=tangentQueryValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, crkState=crkState, crkTangentState=crkTangentState,
        renormalizationState=renormalizationState, renormalizationTangentState=renormalizationTangentState,
        gradientMode=gradientMode, operationMode=operationMode)


def computeSPHInterpolateGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentReferenceValues=None,
        referenceVolumes=None, tangentReferenceVolumes=None, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Interpolate,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, operationMode=operationMode)


def computeSPHLaplacianBrookshawGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentQueryValues=None,
        tangentReferenceValues=None, referenceVolumes=None, tangentReferenceVolumes=None, crkState=None,
        crkTangentState=None, renormalizationState=None, renormalizationTangentState=None,
        gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Laplacian,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues, tangentQueryValues=tangentQueryValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, crkState=crkState, crkTangentState=crkTangentState,
        renormalizationState=renormalizationState, renormalizationTangentState=renormalizationTangentState,
        gradientMode=gradientMode, operationMode=operationMode, laplacianMode=LaplacianScheme.Brookshaw)


def computeSPHLaplacianNaiveGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentQueryValues=None,
        tangentReferenceValues=None, referenceVolumes=None, tangentReferenceVolumes=None,
        gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Laplacian,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues, tangentQueryValues=tangentQueryValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, gradientMode=gradientMode,
        operationMode=operationMode, laplacianMode=LaplacianScheme.Naive)


def computeSPHLaplacianDotGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentQueryValues=None,
        tangentReferenceValues=None, referenceVolumes=None, tangentReferenceVolumes=None, crkState=None,
        crkTangentState=None, renormalizationState=None, renormalizationTangentState=None,
        gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Laplacian,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues, tangentQueryValues=tangentQueryValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, crkState=crkState, crkTangentState=crkTangentState,
        renormalizationState=renormalizationState, renormalizationTangentState=renormalizationTangentState,
        gradientMode=gradientMode, operationMode=operationMode, laplacianMode=LaplacianScheme.Dot)


def computeSPHLaplacianDefaultGeometryJVP(
        queryParticles, domain, kernel, supportMode, adjacency, queryTangentState, referenceParticles=None,
        referenceTangentState=None, queryValues=None, referenceValues=None, tangentQueryValues=None,
        tangentReferenceValues=None, referenceVolumes=None, tangentReferenceVolumes=None, crkState=None,
        crkTangentState=None, renormalizationState=None, renormalizationTangentState=None,
        gradientMode=GradientScheme.Symmetric, operationMode=OperationDirection.AllToAll):
    return _launch(WarpOperation.Laplacian,queryParticles,domain,kernel,supportMode,adjacency,queryTangentState,
                referenceParticles=referenceParticles, referenceTangentState=referenceTangentState,
        queryValues=queryValues, referenceValues=referenceValues, tangentQueryValues=tangentQueryValues,
        tangentReferenceValues=tangentReferenceValues, referenceVolumes=referenceVolumes,
        tangentReferenceVolumes=tangentReferenceVolumes, crkState=crkState, crkTangentState=crkTangentState,
        renormalizationState=renormalizationState, renormalizationTangentState=renormalizationTangentState,
        gradientMode=gradientMode, operationMode=operationMode, laplacianMode=LaplacianScheme.Default)


def computeSPHLaplacianGeometryJVP(queryParticles,domain,kernel,supportMode,adjacency,
        queryTangentState,laplacianMode=LaplacianScheme.Brookshaw,**kwargs):
    return _launch(WarpOperation.Laplacian,queryParticles,domain,kernel,supportMode,adjacency,
                   queryTangentState,laplacianMode=laplacianMode,**kwargs)

__all__=['computeCovarianceGeometryJVP', 'computeSPHCurlGeometryJVP', 'computeSPHDensityGeometryJVP', 'computeSPHDivergenceGeometryJVP', 'computeSPHGradientGeometryJVP', 'computeSPHInterpolateGeometryJVP', 'computeSPHLaplacianBrookshawGeometryJVP', 'computeSPHLaplacianNaiveGeometryJVP', 'computeSPHLaplacianDotGeometryJVP', 'computeSPHLaplacianDefaultGeometryJVP', 'computeSPHLaplacianGeometryJVP']
