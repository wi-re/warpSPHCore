"""Bounded native/Warp parity for the lifecycle and remaining SPH options."""
from dataclasses import replace
import torch
import warp as wp
import warpSPHCore as sph
from warpSPHCore.radiusSearch.native import nt_verlet as native
from warpSPHCore.native.api import sphOperation
from _support.operators import fixture


def test_lifecycle_and_operator_options_match_warp(device):
    wp.init()
    dtype=sph.get_torch_precision()
    q,r,domain,qf,rf=fixture(device,2,1)
    def cast(state):
        return replace(state,**{key:getattr(state,key).to(dtype) for key in ('positions','supports','masses','densities')})
    q,r=cast(q),cast(r)
    domain=replace(domain,min=domain.min.to(dtype),max=domain.max.to(dtype))
    qf,rf=qf.to(dtype),rf.to(dtype)
    mode=sph.SupportScheme.SuperSymmetric
    actual=native.buildVerletList(q,domain,1.4,mode,referenceParticles=r,storeI=False)
    expected=sph.buildVerletList(q,domain,1.4,mode,referenceParticles=r,storeI=False)
    def canonical(graph):
        i=torch.repeat_interleave(torch.arange(graph.numRows,device=device),graph.numNeighbors.long(),output_size=graph.numEdges)
        return torch.sort(i*graph.numCols+graph.paddedJ[:graph.numEdges]).values
    torch.testing.assert_close(canonical(actual),canonical(expected),rtol=0,atol=0)
    from warpSPHCore.radiusSearch.verlet.wp_validity import verletValidityMetrics as warp_metrics
    args=(q.positions+.01,r.positions-.02,actual.queryPositions,actual.referencePositions,
          q.supports,r.supports,actual.querySupports,actual.referenceSupports,
          domain.periodic,domain.min,domain.max,1.4,2)
    for a,b in zip(native.verletValidityMetrics(*args),warp_metrics(*args)):
        torch.testing.assert_close(a,b,rtol=5e-5,atol=2e-6)
    for filtering in (sph.SupportScheme.Gather,sph.SupportScheme.Scatter,mode):
        a=native.filterVerletList(q,domain,actual,filtering,referenceParticles=r,storeI=False)
        b=sph.filterVerletList(q,domain,expected,filtering,referenceParticles=r,storeI=False)
        torch.testing.assert_close(canonical(a),canonical(b),rtol=0,atol=0)
    tolerance=5e-5 if dtype==torch.float32 else 3e-9
    for operation in (sph.WarpOperation.Density,sph.WarpOperation.Gradient,sph.WarpOperation.Laplacian):
        prop=sph.OperationProperties(sph.KernelFunctions.Wendland4,operation=operation,
            gradientMode=sph.GradientScheme.Difference,calibrateNormalization=True,n_h=2.57,
            positiveDivergence=operation==sph.WarpOperation.Laplacian)
        options=dict(referenceParticles=r,adjacency=expected)
        wanted=sph.warpOperation(q,prop,domain,qf,rf,**options)
        got=sphOperation(q,prop,domain,qf,rf,**options)
        torch.testing.assert_close(got,wanted,rtol=tolerance,atol=tolerance)
