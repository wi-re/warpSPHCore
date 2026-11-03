"""Bounded native ownership contracts; multi-process qualification is opt-in."""
from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
import pytest
import torch
import warpSPHCore as sph
from warpSPHCore.mfm import MeshlessNative


def sample(device,dimension=2,dtype=torch.float64,count=17):
    coordinate=torch.arange(count,device=device,dtype=dtype)
    positions=torch.stack([.5*torch.sin(coordinate*(axis+.7)+.1) for axis in range(dimension)],-1)
    supports=torch.ones_like(coordinate)*1.1
    particles=sph.ParticleState(positions,supports,supports*.02,
        torch.zeros(count,device=device,dtype=torch.int32),torch.ones_like(supports))
    domain=sph.DomainDescription(coordinate.new_full((dimension,),-1),coordinate.new_full((dimension,),1),
                                 torch.ones(dimension,device=device,dtype=torch.bool),dimension)
    index_bytes=4 if sph.get_torch_index_dtype()==torch.int32 else 8
    policy=sph.RowChunkPolicy(edgeBudget=150,maxRows=4,bytesPerEdge=index_bytes+8,maxElementsPerEdge=1)
    return particles,domain,policy


def check_rows(adjacency,flat):
    torch.testing.assert_close(adjacency.numNeighbors,flat.numNeighbors)
    for chunk in adjacency:
        local,interval=chunk.adjacency,chunk.metadata
        assert local.paddedJ.numel()==interval.capacity<=adjacency.plan.policy.capacityLimit
        for row in range(interval.rowCount):
            count=int(local.numNeighbors[row]);start=int(local.edgeOffsets[row]);old=int(flat.edgeOffsets[row+interval.rowStart])
            torch.testing.assert_close(local.paddedJ[start:start+count].long(),flat.paddedJ[old:old+count].long())
            if local.paddedI is not None:
                assert bool((local.paddedI[start:start+count]==row+interval.rowStart).all())
        assert bool((local.paddedJ[interval.liveEdges:]==-1).all())


def test_bounded_search_verlet_checkpoint_and_storage_lifetimes(device,monkeypatch):
    for dimension,dtype in ((1,torch.float32),(2,torch.float64),(3,torch.float32)):
        particles,domain,policy=sample(device,dimension,dtype)
        flat=sph.radiusSearch(particles,domain,mode=sph.SupportScheme.SuperSymmetric)
        library=sph.build()
        # A chunked rebuild must count/collect directly into owned buffers.
        with monkeypatch.context() as patch:
            patch.setattr(library,'radius',lambda *args:pytest.fail('allocated a global neighbor list'))
            chunks=sph.radiusSearchRowChunks(particles,domain,policy=policy,storeI=True,
                                             mode=sph.SupportScheme.SuperSymmetric)
            check_rows(chunks,flat)
            retained=sph.buildChunkedVerletList(particles,domain,policy=policy,verletScale=1.2)
            assert sph.buildChunkedVerletList(particles,domain,policy=policy,verletScale=1.2,
                                              priorNeighborhood=retained) is retained
            filtered=sph.filterChunkedVerletList(particles,domain,retained,policy=replace(policy,maxRows=2))
            check_rows(filtered,flat)
            restored=sph.restoreRowChunkCheckpointState(sph.rowChunkCheckpointState(retained),
                device=device,policy=replace(policy,maxRows=2))
            check_rows(sph.filterChunkedVerletList(particles,domain,restored),flat)
            sph.radiusSearchRowChunks(particles,domain,policy=policy,reuse=chunks,inPlace=True,
                                     storeI=True,mode=sph.SupportScheme.SuperSymmetric)
            with pytest.raises(ValueError,match='overwritten'):chunks.assertStorageValid()


def test_owned_operator_corrections_polynomial_and_meshless_pipeline(device):
    particles,domain,policy=sample(device)
    chunks=sph.radiusSearchRowChunks(particles,domain,policy=policy,mode=sph.SupportScheme.SuperSymmetric)
    flat=sph.radiusSearch(particles,domain,mode=sph.SupportScheme.SuperSymmetric)
    properties=sph.OperationProperties(sph.KernelFunctions.Wendland2,gradientMode=sph.GradientScheme.Difference)
    prepared=sph.prepareCRK(particles,domain,properties.kernel,adjacency=chunks)
    reference=sph.prepareCRK(particles,domain,properties.kernel,adjacency=flat)
    torch.testing.assert_close(prepared.crkState.A,reference.crkState.A,rtol=1e-10,atol=1e-10)
    correction=sph.computeRenormalizationMatrices(particles,properties,domain,adjacency=chunks,returnEigVals=False)
    for operation in (sph.WarpOperation.Density,sph.WarpOperation.Interpolate,sph.WarpOperation.Gradient,
                      sph.WarpOperation.Divergence,sph.WarpOperation.Curl,sph.WarpOperation.Laplacian,sph.WarpOperation.Covariance):
        props=replace(properties,operation=operation)
        values=particles.positions.sin()
        options=dict(referenceVolumes=prepared.referenceVolumes,crkState=prepared.crkState,
            renormalizationState=correction,gradHState=sph.GradHState(particles.densities*.9),
            covarianceReturnNumNeighbors=True)
        actual=sph.warpOperation(particles,props,domain,values,adjacency=chunks,**options)
        expected=sph.warpOperation(particles,props,domain,values,adjacency=flat,**options)
        if not isinstance(actual,tuple):actual,expected=(actual,),(expected,)
        for left,right in zip(actual,expected):torch.testing.assert_close(left,right,rtol=1e-9,atol=1e-9)
    # Equal population sizes do not imply aliases: keep global reference roles
    # distinct from the sliced queries, including symmetric density factors.
    reference=replace(particles,supports=particles.supports*torch.linspace(.7,1.1,17,device=device),
        kinds=torch.arange(17,device=device,dtype=torch.int32)%3,
        densities=particles.densities*1.3)
    for count in (17,11):
        query=replace(particles,positions=particles.positions[:count]+.03,
            supports=particles.supports[:count]*.9,masses=particles.masses[:count],
            densities=particles.densities[:count]*.8,kinds=reference.kinds[:count].flip(0))
        owned=sph.radiusSearchRowChunks(query,domain,referenceParticles=reference,policy=policy,
                                       mode=sph.SupportScheme.SuperSymmetric)
        whole=sph.radiusSearch(query,domain,referenceParticles=reference,mode=sph.SupportScheme.SuperSymmetric)
        props=replace(properties,operation=sph.WarpOperation.Gradient,
            gradientMode=sph.GradientScheme.Symmetric,operationMode=sph.OperationDirection.FluidToAll)
        options=dict(referenceParticles=reference,queryValues=query.positions.cos(),
            referenceValues=reference.positions.sin(),queryVolumes=query.masses/query.densities,
            referenceVolumes=reference.masses/reference.densities,
            gradHState=sph.GradHState(query.densities*.9,reference.densities*.8))
        torch.testing.assert_close(sph.warpOperation(query,props,domain,adjacency=owned,**options),
            sph.warpOperation(query,props,domain,adjacency=whole,**options),rtol=1e-9,atol=1e-9)
    for constant in (False,True):
        a=sph.PolyFit.build(particles,domain,properties.kernel,2,constant=constant,adjacency=chunks,sector=0,unitWeights=True)
        b=sph.PolyFit.build(particles,domain,properties.kernel,2,constant=constant,adjacency=flat,sector=0,unitWeights=True)
        torch.testing.assert_close(a.coefficients(particles.positions.square()),b.coefficients(particles.positions.square()),rtol=1e-9,atol=1e-9)
    for solver in ('cg','mg'):
        a=MeshlessNative.build(particles.positions,particles.supports,domain,properties.kernel,
            adjacency=chunks,closure_solver=solver,mg_max_coarse=4,cg_iters=50)
        b=MeshlessNative.build(particles.positions,particles.supports,domain,properties.kernel,
            adjacency=flat,closure_solver=solver,mg_max_coarse=4,cg_iters=50)
        torch.testing.assert_close(a.rates(particles.densities,particles.positions*.1,particles.densities,1.4),
                                   b.rates(particles.densities,particles.positions*.1,particles.densities,1.4),rtol=1e-9,atol=1e-9)


def test_owned_empty_rows_reject_ad_and_use_current_stream(device):
    for query_count,reference_count in ((0,5),(3,0),(0,0)):
        query,domain,policy=sample(device,count=query_count)
        reference,_,_=sample(device,count=reference_count)
        adjacency=sph.radiusSearchRowChunks(query,domain,referenceParticles=reference,policy=policy)
        output=sph.warpOperation(query,sph.OperationProperties(sph.KernelFunctions.Wendland2,
            operation=sph.WarpOperation.Density),domain,referenceParticles=reference,adjacency=adjacency)
        torch.testing.assert_close(output,query.supports.new_zeros(query_count))
    particles,domain,policy=sample(device)
    adjacency=sph.radiusSearchRowChunks(particles,domain,policy=policy)
    moving=replace(particles,positions=particles.positions.clone().requires_grad_())
    with pytest.raises(NotImplementedError,match='AD'):
        sph.warpOperation(moving,sph.OperationProperties(sph.KernelFunctions.Wendland2),domain,
                          particles.masses,adjacency=adjacency)
    if device!='cpu':
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            adjacency=sph.radiusSearchRowChunks(particles,domain,policy=policy)
            result=sph.warpOperation(particles,sph.OperationProperties(sph.KernelFunctions.Wendland2),
                                     domain,particles.masses,adjacency=adjacency)
        torch.cuda.current_stream().wait_stream(stream)
        flat=sph.radiusSearch(particles,domain)
        expected=sph.warpOperation(particles,sph.OperationProperties(sph.KernelFunctions.Wendland2),domain,particles.masses,adjacency=flat)
        torch.testing.assert_close(result,expected)


@pytest.mark.cpu
@pytest.mark.extended
def test_two_rank_native_owned_rows_and_coordinated_failures(tmp_path):
    worker=Path(__file__).resolve().parents[2]/'_support/native_row_worker.py'
    environment=dict(os.environ,OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(worker)]
    result=subprocess.run(command,env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    (tmp_path/'gloo.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    assert result.stdout.count('owned-row integration passed')==4,result.stdout
