"""Shared, explicit workloads for operator correctness, resources and timing.

No device work at import. Local matrix stages have no interaction denominator.
All pair counts refer to directed non-self physical pairs on equal-support,
periodic, all-fluid lattices, with support less than half the box extent.
"""
from dataclasses import asdict, dataclass, replace
from itertools import product

CORE = ('density', 'interpolate', 'gradient', 'divergence', 'curl', 'laplacian', 'covariance')
PAIR_STAGES = ('crk_volume', 'crk_moments', 'crk_density', 'renorm', 'crk_factors')
LOCAL = ('pinv_symmetric', 'apply_pinv', 'pinv_policy', 'torch_pinv', 'crk_terms')
OPERATIONS = CORE + PAIR_STAGES + LOCAL


@dataclass(frozen=True)
class Case:
    n: int
    dim: int = 2
    neighbors: int = 55
    operation: str = 'gradient'
    traversal: str = 'adjacency'
    lanes: int = 1
    form: str = 'plain'
    kernel: str = 'Wendland2'
    support: str = 'Gather'
    gradient: str = 'Difference'
    laplacian: str = 'Brookshaw'

    def __post_init__(self):
        side = round(self.n ** (1 / self.dim)) if self.n > 0 and self.dim in (2, 3) else 0
        if side < 1 or side ** self.dim != self.n:
            raise ValueError('N must be an exact positive dim-th power, dim=2 or 3')
        if self.neighbors <= 0 or self.operation not in OPERATIONS:
            raise ValueError('invalid neighbors or operation')
        if self.traversal not in ('adjacency', 'grid', 'rebuild') or self.lanes not in (1, 2, 4, 8, 16, 32):
            raise ValueError('invalid traversal or lanes')
        if self.form not in ('plain', 'renorm', 'crk'):
            raise ValueError('invalid form')

    def dictionary(self):
        return asdict(self)


def cases(ns, *, dim=2, neighbors=(25, 55, 110, 220), operations=OPERATIONS,
          traversals=('adjacency', 'grid'), lanes=(1, 32), forms=('plain',),
          kernels=('Wendland2',), supports=('Gather',), gradients=('Difference',), laplacians=('Brookshaw',)):
    """Deduplicate irrelevant options, retaining meaningful names in each row."""
    seen = set()
    for n, degree, op, traversal, lane, form, kernel, support, grad, lap in product(
            ns, neighbors, operations, traversals, lanes, forms, kernels, supports, gradients, laplacians):
        if form != 'plain' and op not in ('gradient', 'divergence', 'curl', 'laplacian'):
            continue
        c = Case(n, dim, degree if op not in LOCAL else 55, op,
                 traversal if op not in LOCAL else 'adjacency', lane if op not in LOCAL else 1,
                 form, kernel if op not in LOCAL else 'Wendland2',
                 support if op in CORE or op == 'renorm' else 'Gather',
                 grad if op in ('gradient', 'divergence', 'curl') else 'Difference',
                 lap if op == 'laplacian' else 'Brookshaw')
        if c not in seen:
            seen.add(c)
            yield c


def work_counts(case, edges):
    local = case.operation in LOCAL
    passes = 3 if case.operation == 'crk_factors' else 1
    return dict(pair_passes=0 if local else passes,
                evaluated_nonself_pairs=None if local else passes * (edges - case.n),
                candidate_entries=None if local or case.traversal != 'adjacency' else passes * edges,
                matrices=case.n if local or case.operation in ('renorm', 'crk_factors') else None)


def tensors(value):
    import torch
    from dataclasses import fields, is_dataclass
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif is_dataclass(value):
        for f in fields(value):
            yield from tensors(getattr(value, f.name))


def check_output(value, reference=None):
    import torch
    out = list(tensors(value))
    if not out or not all(bool(torch.isfinite(t).all()) for t in out):
        raise AssertionError('missing or nonfinite output')
    if reference is not None:
        refs = list(tensors(reference))
        if len(out) != len(refs):
            raise AssertionError('output arity differs')
        for actual, expected in zip(out, refs):
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError('output shape or dtype differs')
            # Cancellation near zero requires an absolute tolerance as well.
            tol = 2e-10 if actual.dtype == torch.float64 else 2e-4
            torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


class Workload:
    """Fixed geometry; topology/correction construction is outside timing.

    Rebuild traversal explicitly passes None and includes public implicit grid
    builds. CRK stages consume frozen production factors/moments; the full
    factor call builds them itself. Matrices are nontrivial SPD batches with
    every seventh row rank deficient, deterministic and independent of topology.
    """
    def __init__(self, n, dim, neighbors, device):
        import torch
        from warpSPHCore import ParticleState, radiusSearchCompactHashMap, setNeighborLanes
        from warpSPHCore.enumTypes import SupportScheme
        from warpSPHCore.util import generateNeighborTestData
        p, h, _, self.domain, dx = generateNeighborTestData(round(n ** (1 / dim)), neighbors, dim, True, device)
        if len(p) != n or float(h.max()) >= 1:
            raise ValueError('lattice size mismatch or support >= half periodic box; increase N')
        g = torch.Generator(device=device).manual_seed(173)
        self.scalar = torch.randn(n, dtype=p.dtype, device=device, generator=g)
        self.vector = torch.randn(n, dim, dtype=p.dtype, device=device, generator=g)
        self.particles = ParticleState(positions=p, supports=h, masses=torch.full_like(self.scalar, dx ** dim),
                                       densities=torch.ones_like(self.scalar), kinds=torch.zeros(n, dtype=torch.int32, device=device))
        setNeighborLanes(1)
        self.adjacency, self.grid = radiusSearchCompactHashMap(self.particles, self.domain, mode=SupportScheme.Gather,
                                                              returnCompactHashMap=True)
        self.edges = int(self.adjacency.numNeighbors.sum())
        self.n, self.dim = n, dim
        a = torch.randn(n, dim, dim, device=device, dtype=p.dtype, generator=g)
        self.matrix = a @ a.mT + .25 * torch.eye(dim, device=device, dtype=p.dtype)
        self.matrix[::7, -1, :] = 0
        self.matrix[::7, :, -1] = 0
        self._cache = {}

    def prepared(self, kernel, *, need_renorm=False):
        import torch
        from warpSPHCore import OperationProperties, setNeighborLanes
        from warpSPHCore.crk import computeCRKFactors
        from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
        from warpSPHCore.enumTypes import KernelFunctions, SupportScheme
        from warpSPHCore.renorm import computeRenormalizationMatrices
        if kernel not in self._cache:
            setNeighborLanes(1)
            k = getattr(KernelFunctions, kernel)
            area, density, crk = computeCRKFactors(self.particles, self.domain, k, adjacency=self.adjacency)
            props = OperationProperties(kernel=k, supportMode=SupportScheme.Scatter)
            moments = _computeCRKMoments_stateBackend(self.particles, props, self.domain, adjacency=self.adjacency,
                                                      queryVolumes=area, referenceVolumes=area)
            renorm = None
            check_output((area, density, crk, moments, renorm))
            self._cache[kernel] = area, crk, moments, renorm
        if need_renorm and self._cache[kernel][3] is None:
            area, crk, moments, _ = self._cache[kernel]
            renorm = computeRenormalizationMatrices(self.particles, OperationProperties(kernel=getattr(KernelFunctions, kernel)),
                                                     self.domain, adjacency=self.adjacency, returnEigVals=False)
            check_output(renorm)
            self._cache[kernel] = area, crk, moments, renorm
        return self._cache[kernel]

    def call(self, c):
        import torch
        from warpSPHCore import OperationProperties, warpOperation
        from warpSPHCore.crk import computeCRKFactors
        from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend
        from warpSPHCore.crk.crk_moments import _computeCRKMoments_stateBackend
        from warpSPHCore.crk.crk_density import _computeCRKDensity_stateBackend
        from warpSPHCore.crk.crk_terms import computeCRKTermsWarp
        from warpSPHCore.enumTypes import KernelFunctions, SupportScheme, GradientScheme, LaplacianScheme, WarpOperation
        from warpSPHCore.linalg import pinvSymmetric, applyPinv
        from warpSPHCore.pinv import pinv_warp
        from warpSPHCore.renorm import computeRenormalizationMatrices
        k = getattr(KernelFunctions, c.kernel)
        topo = {'adjacency': self.adjacency, 'grid': self.grid, 'rebuild': None}[c.traversal]
        props = OperationProperties(kernel=k, supportMode=getattr(SupportScheme, c.support),
                                    gradientMode=getattr(GradientScheme, c.gradient), laplacianMode=getattr(LaplacianScheme, c.laplacian))
        p, d = self.particles, self.domain
        if c.operation == 'pinv_symmetric':
            return lambda: pinvSymmetric(self.matrix, rtol=1e-6)
        if c.operation == 'apply_pinv':
            return lambda: applyPinv(self.vector[:, None, :], self.matrix, rtol=1e-6)
        if c.operation == 'pinv_policy':
            return lambda: pinv_warp(self.matrix, self.adjacency.numNeighbors, symmetric=True)
        if c.operation == 'torch_pinv':
            return lambda: torch.linalg.pinv(self.matrix, rtol=1e-6)
        if c.operation == 'renorm':
            return lambda: computeRenormalizationMatrices(p, props, d, adjacency=topo)
        if c.operation == 'crk_factors':
            return lambda: computeCRKFactors(p, d, k, adjacency=topo)
        if c.operation == 'crk_volume':
            return lambda: _computeCRKVolume_stateBackend(p, props, d, adjacency=topo)
        if c.operation.startswith('crk_') or c.form != 'plain':
            area, crk, moments, renorm = self.prepared(c.kernel, need_renorm=c.form == 'renorm')
        if c.operation == 'crk_terms':
            return lambda: computeCRKTermsWarp(*moments[:6], num_nbrs=moments[6], supports=p.supports)
        if c.operation == 'crk_moments':
            props = replace(props, supportMode=SupportScheme.Scatter)
            return lambda: _computeCRKMoments_stateBackend(p, props, d, adjacency=topo, queryVolumes=area, referenceVolumes=area)
        if c.operation == 'crk_density':
            props = replace(props, supportMode=SupportScheme.Scatter)
            return lambda: _computeCRKDensity_stateBackend(p, props, d, adjacency=topo, queryVolumes=area, referenceVolumes=area, crkState=crk)
        props = replace(props, operation=getattr(WarpOperation, c.operation.capitalize()))
        values = self.vector if c.operation in ('divergence', 'curl') or (c.operation == 'laplacian' and c.laplacian == 'Dot') else self.scalar
        correction = {'renormalizationState': renorm} if c.form == 'renorm' else {'crkState': crk} if c.form == 'crk' else {}
        return lambda: warpOperation(p, props, d, adjacency=topo, queryValues=values, referenceValues=values, **correction)
