"""Eager particle-output operators over resident, whole-row CSR chunks.

References keep global IDs. Every query-role array is sliced together; tensor
extras declare their role rather than guessing it from equal population sizes.
Only O(N) particle outputs are assembled, by disjoint writes, never pair data.
"""
from dataclasses import fields, is_dataclass, replace

import torch
import torch.autograd.forward_ad as fwAD

from ..dataTypes import CRKState, GradHState, ParticleState, RenormalizationState, ExecutionMode
from ..radiusSearch.chunked import _deferred
from .._backend import BACKEND
from ..radiusSearch.pairField import ChunkedPairField
def _allocate_native_output(shape,dtype,device):
    dtype,trailing = dtype if isinstance(dtype,tuple) else (dtype,())
    return torch.empty((shape,*trailing),dtype=dtype,device=device)

if BACKEND == 'warp':
    from .launcher import _allocate_output

from .lanes import neighborLanes
from .operator_spec import Corrections, ExtraKind, RowRole, ShapeOf, SPHContext


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, ChunkedPairField):
        for tensor in value.buffers:
            yield from _tensors(tensor)
    elif is_dataclass(value):
        for f in fields(value):
            yield from _tensors(getattr(value, f.name))
    elif isinstance(value, (tuple, list, dict)):
        for v in value.values() if isinstance(value, dict) else value:
            yield from _tensors(v)


def _inert(values):
    if any(t.requires_grad or fwAD.unpack_dual(t)[1] is not None for t in _tensors(values)):
        raise NotImplementedError('Row-chunk operators do not support reverse or forward AD.')


def _field(value, rows, device, name):
    if value is not None and (not isinstance(value, torch.Tensor) or value.ndim == 0
                              or value.shape[0] != rows or value.device != device):
        raise ValueError(f'Row-chunk {name} requires {rows} particle rows on {device}.')
    return value


def _particle(p, rows, device, name, interval=None):
    values = {}
    for name_ in ('positions', 'supports', 'masses', 'kinds', 'densities'):
        value = _field(getattr(p, name_), rows, device, f'{name}.{name_}')
        values[name_] = value[interval] if value is not None and interval is not None else value
    return ParticleState(**values)


def validateContext(ctx):
    adjacency = ctx.adjacency
    adjacency.assertStorageValid()
    if adjacency.communication is not None:
        adjacency.communication.validateAdjacency(adjacency)
    elif adjacency.plan.numOwners != 1:
        raise NotImplementedError('Partial row-chunk outputs require a complete single-owner list or an explicit distributed process group.')
    if ctx.mode in (ExecutionMode.FORWARD, ExecutionMode.REVERSE):
        raise NotImplementedError('Row-chunk operators support eager execution without AD only.')
    device = ctx.query.positions.device
    if _deferred() is not None or (device.type == 'cuda' and torch.cuda.is_current_stream_capturing()):
        raise NotImplementedError('Row-chunk operators do not support deferred/graph capture.')
    if device != adjacency.queryPositions.device:
        raise ValueError('Row-chunk adjacency and particle fields must share a device.')
    reference = ctx.query if ctx.reference is None else ctx.reference
    queryFields = _particle(ctx.query, adjacency.numRows, device, 'query')
    referenceFields = _particle(reference, adjacency.numCols, device, 'reference')
    _inert((queryFields, referenceFields, ctx.domain, ctx.corrections))
    # Check correction row identity even for empty populations, before allocation.
    _corrections(ctx, slice(0, 0))


def _corrections(ctx, interval):
    adjacency, c = ctx.adjacency, ctx.corrections
    nq, nr, device = adjacency.numRows, adjacency.numCols, ctx.query.positions.device
    def query(value, name):
        value = _field(value, nq, device, name)
        return value[interval] if value is not None else None
    def reference(value, name):
        return _field(value, nr, device, name)
    qvol, rvol = c.volumes
    # Resolve aliases BEFORE slicing; global J must never read a local fallback.
    rvol = qvol if rvol is None else rvol
    crk = None
    if c.crk is not None:
        qc, rc = c.crk if isinstance(c.crk, tuple) else (c.crk, c.crk)
        # Some specs (CRK consistency density) read query corrections only.
        # Reference CRK rows are validated against the spec before any launch.
        crk = (CRKState(**{f.name: query(getattr(qc, f.name), 'query CRK.' + f.name) for f in fields(CRKState)}), rc)
    gradH = None
    if c.gradH is not None:
        if not isinstance(c.gradH, GradHState):
            raise TypeError('Row-chunk contexts require GradHState for grad-H corrections.')
        qo = c.gradH.queryOmegas
        ro = qo if c.gradH.referenceOmegas is None else c.gradH.referenceOmegas
        gradH = GradHState(query(qo, 'query omega'), reference(ro, 'reference omega'))
    renorm = None if c.renorm is None else RenormalizationState(query(c.renorm.renormalizationMatrices, 'renormalization'))
    return Corrections((query(qvol, 'query volume'), reference(rvol, 'reference volume')), crk, gradH, renorm)


def chunkContexts(ctx):
    """Private local-row contexts; optional stored I remains global and unused."""
    adjacency = ctx.adjacency
    reference = ctx.query if ctx.reference is None else ctx.reference
    for chunk in adjacency:
        m = chunk.metadata
        interval = slice(m.rowStart, m.rowStop)
        yield interval, SPHContext(
            query=_particle(ctx.query, adjacency.numRows, ctx.query.positions.device, 'query', interval),
            reference=reference, properties=ctx.properties, domain=ctx.domain,
            adjacency=chunk.adjacency, corrections=_corrections(ctx, interval), mode=ctx.mode)


def _validateSpec(spec, ctx, extras):
    if (not spec.rowChunked or not spec.outputs or spec.outputs[0].shape is not ShapeOf.QUERY
        or spec.outputs[0].chunkScratch or spec.numThreads is not None or
        any(not (o.shape is ShapeOf.QUERY and not o.chunkScratch) and
            not (o.chunkScratch and type(o.shape) is int and o.shape == 1) for o in spec.outputs)):
        raise NotImplementedError('This operator has not opted into row-chunk particle outputs.')
    declared = {e.name: e for e in spec.extras}
    if extras.keys() != declared.keys():
        raise TypeError('Row-chunk extras must match the declared operator arguments.')
    _inert(extras)
    if ctx.corrections.crk is not None and not spec.queryOnlyCRK:
        c = ctx.corrections.crk
        reference = c[1] if isinstance(c, tuple) else c
        for f in fields(CRKState):
            _field(getattr(reference, f.name), ctx.adjacency.numCols,
                   ctx.query.positions.device, 'reference CRK.' + f.name)
    roles = {name: e.resolveRowRole(ctx, extras) for name, e in declared.items()}
    for name, e in declared.items():
        value = extras[name]
        role = roles[name]
        if e.kind is ExtraKind.CHUNK_TENSOR and role is RowRole.CHUNK:
            if not isinstance(value, ChunkedPairField):
                raise TypeError(f'Extra {name} requires an owned ChunkedPairField.')
            value.validate(ctx.adjacency)
        elif e.kind in (ExtraKind.TENSOR, ExtraKind.CHUNK_TENSOR):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'Extra {name} must be a tensor.')
            if value.device != ctx.query.positions.device:
                raise ValueError(f'Row-chunk tensor extra {name} must share the particle device.')
            if role not in (RowRole.QUERY, RowRole.REFERENCE, RowRole.REPLICATED):
                raise NotImplementedError(f'Tensor extra {name} needs an explicit row role for chunk dispatch.')
            if role is not RowRole.REPLICATED:
                _field(value, ctx.adjacency.numRows if role is RowRole.QUERY else ctx.adjacency.numCols,
                       ctx.query.positions.device, name)
        elif isinstance(value, torch.Tensor):
            raise TypeError(f'Extra {name} must be a scalar.')
        elif role is RowRole.ROW_START:
            if type(value) is not int or value != 0:
                raise ValueError('The chunk ROW_START placeholder must be integer zero; dispatch supplies the global row base.')
        elif role is not None:
            raise ValueError(f'Scalar extra {name} cannot declare a particle row role.')
    return roles


def launchRowChunks(spec, ctx, extras, prepared=None):
    """Complete owned rows, then replicate particle results before dependent reads."""
    if spec.chunkedNeighborLanes is not None:
        spec = replace(spec, defaultNeighborLanes=spec.chunkedNeighborLanes, chunkedNeighborLanes=None)
    communication = ctx.adjacency.communication
    if communication is None:
        return _launchLocal(spec, ctx, extras, prepared)
    def validate():
        validateContext(ctx)
        _validateSpec(spec, ctx, extras)
        scalars = {e.name: str(extras[e.name]) for e in spec.extras
                   if e.kind not in (ExtraKind.TENSOR, ExtraKind.CHUNK_TENSOR)}
        tensors = {e.name: (list(extras[e.name].shape), str(extras[e.name].dtype))
                   for e in spec.extras if isinstance(extras[e.name], torch.Tensor)}
        outputs = [(out.resolve(ctx, extras)[0], str(out.resolve(ctx, extras)[1]), out.chunkScratch)
                   for out in spec.outputs]
        # Match operator_spec's effective selection, including scalar fallback.
        # Different reduction trees must not silently mix across owners.
        kernel,lanes=spec.kernel,1
        if spec.tiledKernel is not None and ctx.query.positions.is_cuda:
            lanes=neighborLanes(default=spec.defaultNeighborLanes)
            if lanes>1:kernel=spec.tiledKernel
        return (ctx.adjacency.plan.signature, spec.kernel.key, str(ctx.properties), scalars, tensors, outputs,
                (kernel.key, lanes, (lanes if lanes>1 else (spec.blockDim or 256))
                 if ctx.query.positions.is_cuda else 1))
    communication._checkedAgree('operator:contract',validate)
    # Exact mode fingerprints every reference input. Contract mode checks layouts
    # here and retains exact checks at search/accepted/checkpoint boundaries.
    # Owned pair buffers and unread scalar dummies are excluded.
    reference = ctx.query if ctx.reference is None else ctx.reference
    inputs = {str(i): t for i, t in enumerate(_tensors((ctx.query, reference, ctx.domain, ctx.corrections)))}
    for e in spec.extras:
        role = e.resolveRowRole(ctx, extras)
        if role in (RowRole.QUERY, RowRole.REFERENCE) and isinstance(extras[e.name], torch.Tensor):
            inputs['extra:' + e.name] = extras[e.name]
    communication.assertReplicated('operator:replicated-inputs', inputs)
    result = communication.localPhase('operator:compute', lambda: _launchLocal(spec, ctx, extras, prepared))
    outputs = (result,) if len(spec.outputs) == 1 else result
    communication.exchangeParticleOutputs(ctx.adjacency,
        tuple(t for o, t in zip(spec.outputs, outputs) if not o.chunkScratch), tag='operator:' + spec.kernel.key)
    return result


def _launchLocal(spec, ctx, extras, prepared=None):
    """Run local chunks only; unowned particle rows are unread until exchange."""
    from .operator_spec import launchOperator
    validateContext(ctx)
    roles = _validateSpec(spec, ctx, extras)
    outputs = []
    for out in spec.outputs:
        shape, dtype = out.resolve(ctx, extras)
        if hasattr(spec.kernel,'launch_native'):
            tensor = _allocate_native_output(shape,dtype,ctx.query.positions.device)
        else:
            tensor, _ = _allocate_output(shape,dtype,str(ctx.query.positions.device),False)
            del tensor._warp_array
        outputs.append(tensor)
    contexts = chunkContexts(ctx) if prepared is None else prepared
    for index, (interval, local) in enumerate(contexts):
        localExtras = {}
        for e in spec.extras:
            role, value = roles[e.name], extras[e.name]
            localExtras[e.name] = (value[interval] if role is RowRole.QUERY else
                                  value.buffers[index] if role is RowRole.CHUNK else
                                  interval.start if role is RowRole.ROW_START else value)
        result = launchOperator(spec, local, **localExtras) if prepared is None else local.launch(spec, **localExtras)
        localOutputs = (result,) if len(outputs) == 1 else result
        for out, target, source in zip(spec.outputs, outputs, localOutputs):
            if not out.chunkScratch:
                target[interval].copy_(source)
        del localOutputs, result, source
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
