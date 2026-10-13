"""Opt-in call audit: which named section issues how many torch ops, warp kernel
launches and host syncs.

Off by default and free when off (`profiling.record_function` returns a
`nullcontext`). Switch it on with `WARPSPHCORE_AUDIT=1` (a report is printed at
exit) or `callAudit.enable()`; every `record_function("...")` call site already in
warpSPH / warpSPHCore then becomes an audited section, so no code is edited to
instrument a module call. `markStep()` (called once per step by the runner) splits the
counts per step so the report can name the steps that spike.

What is counted, per section (inclusive of nested sections, plus a `self` figure that
excludes them):

* **ops**       every aten op dispatched (`TorchDispatchMode`): the "thousands of tiny
                torch calls" number. Under a CUDA-graph *capture* these are the ops
                recorded; a replay issues none.
* **launches**  warp kernels launched through `wp.launch` / `wp.launch_tiled`.
* **syncs**     host synchronisations, with the Python call site: the ones PyTorch's
                sync-debug mode sees (`.item()`, `.nonzero()`, `.cpu()`, a blocking
                copy, ...) plus explicit `torch.cuda.synchronize()` and
                `wp.synchronize*()`. PyTorch's mode is a prototype and misses some
                (e.g. a warp `array.numpy()`), so a zero here is "none seen", not a proof.
* **replays**   CUDA-graph replays.

    WARPSPHCORE_AUDIT=1 python scripts/probe_contactLine.py ...        # report at exit
    callAudit.enable(); ...; print(callAudit.report()); print(callAudit.stepReport())

Stdlib-only at import (the profiling module imports this before anything else);
torch and warp are imported lazily by `enable`.
"""

import atexit
import collections
import os
import sys
import threading
import traceback
import warnings
from contextlib import contextmanager

enabled = False

_SYNC_MESSAGE = 'called a synchronizing CUDA operation'
_lock = threading.RLock()


class _Agg:
    """Counts of one section name over the run."""
    __slots__ = ('calls', 'ops', 'launches', 'syncs', 'replays', 'selfOps', 'selfLaunches',
                 'selfSyncs', 'opNames', 'kernelNames', 'syncSites', 'parents')

    def __init__(self):
        self.calls = self.ops = self.launches = self.syncs = self.replays = 0
        self.selfOps = self.selfLaunches = self.selfSyncs = 0
        self.opNames = collections.Counter()
        self.kernelNames = collections.Counter()
        self.syncSites = collections.Counter()
        self.parents = collections.Counter()    # enclosing section name (None at top level) -> entries


_sections = collections.OrderedDict()           # name -> _Agg
_open = []                                      # stack of (name, _Agg)
_stepNow = collections.defaultdict(lambda: [0, 0, 0])   # name -> [ops, launches, syncs] this step
_stepHistory = collections.deque(maxlen=400)    # [(stepIndex, {name: (ops, launches, syncs)})]
_stepIndex = 0
_outside = _Agg()                               # events with no open section
_total = _Agg()                                 # every event, in or out of a section
_options = {}
_patched = []                                   # (obj, attr, original) to undo
_dispatchMode = None
_origShowWarning = None


def _addTo(agg, kind, name, site):
    if kind == 'op':
        agg.ops += 1
    elif kind == 'launch':
        agg.launches += 1
    elif kind == 'sync':
        agg.syncs += 1
    elif kind == 'replay':
        agg.replays += 1


def _record(kind, name, site=None):
    """Add one event to every open section (once per distinct name) and to the innermost."""
    with _lock:
        _addTo(_total, kind, name, site)
        if not _open:
            targets = [(None, _outside)]
        else:
            seen, targets = set(), []
            for secName, agg in _open:
                if secName not in seen:
                    seen.add(secName)
                    targets.append((secName, agg))
        for secName, agg in targets:
            if kind == 'op':
                agg.ops += 1
                agg.opNames[name] += 1
                if secName is not None:
                    _stepNow[secName][0] += 1
            elif kind == 'launch':
                agg.launches += 1
                agg.kernelNames[name] += 1
                if secName is not None:
                    _stepNow[secName][1] += 1
            elif kind == 'sync':
                agg.syncs += 1
                agg.syncSites[site or name] += 1
                if secName is not None:
                    _stepNow[secName][2] += 1
            elif kind == 'replay':
                agg.replays += 1
        if _open:
            inner = _open[-1][1]
            if kind == 'op':
                inner.selfOps += 1
            elif kind == 'launch':
                inner.selfLaunches += 1
            elif kind == 'sync':
                inner.selfSyncs += 1


def _callSite():
    """First frame outside torch / warp / this module: where the sync was asked for."""
    for fr in reversed(traceback.extract_stack(limit=25)):
        fn = fr.filename
        if fn == __file__ or '/torch/' in fn or '/warp/' in fn or '/contextlib' in fn:
            continue
        return f"{os.path.basename(fn)}:{fr.lineno} {fr.name}"
    return '?'


@contextmanager
def section(name, inner=None):
    """Audit one named section; nests. `inner` is an optional context manager entered
    with it (the torch profiler's `record_function` when `WARPSPHCORE_PROFILING=1`)."""
    with _lock:
        agg = _sections.get(name)
        if agg is None:
            agg = _sections[name] = _Agg()
        agg.calls += 1
        agg.parents[_open[-1][0] if _open else None] += 1
        _open.append((name, agg))
    try:
        if inner is None:
            yield
        else:
            with inner:
                yield
    finally:
        with _lock:
            _open.pop()


def markStep():
    """Close the current step's counts (call once per step; the runner does)."""
    global _stepIndex
    if not enabled:
        return
    with _lock:
        _stepHistory.append((_stepIndex, {k: tuple(v) for k, v in _stepNow.items()}))
        _stepNow.clear()
        _stepIndex += 1


def reset():
    """Forget every count (e.g. after warm-up and graph capture, before the measured steps)."""
    global _stepIndex
    with _lock:
        _sections.clear()
        _open.clear()
        _stepNow.clear()
        _stepHistory.clear()
        _stepIndex = 0
        _outside.__init__()
        _total.__init__()


# -- the hooks ---------------------------------------------------------------------

def enable(torchOps=True, syncs=True, launches=True):
    """Install the counters. Idempotent. `torchOps=False` drops the (slowest) aten-op
    counter when only syncs and launches matter."""
    global enabled, _dispatchMode, _origShowWarning
    if enabled:
        return
    import torch
    _options.update(torchOps=torchOps, syncs=syncs, launches=launches)

    if torchOps:
        from torch.utils._python_dispatch import TorchDispatchMode

        class _OpCounter(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                _record('op', func.overloadpacket.__name__)
                return func(*args, **(kwargs or {}))

        _dispatchMode = _OpCounter()
        _dispatchMode.__enter__()

    if syncs and torch.cuda.is_available():
        warnings.filterwarnings('always', message=_SYNC_MESSAGE)
        _origShowWarning = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if _SYNC_MESSAGE in str(message):
                _record('sync', 'sync', _callSite())
            else:
                _origShowWarning(message, category, filename, lineno, file, line)
        warnings.showwarning = show
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # the "prototype feature" notice
            torch.cuda.set_sync_debug_mode('warn')

        def wrap(obj, attr, label):
            orig = getattr(obj, attr, None)
            if orig is None:
                return

            def counted(*a, **k):
                _record('sync', label, _callSite() + f' [{label}]')
                return orig(*a, **k)
            _patched.append((obj, attr, orig))
            setattr(obj, attr, counted)
        wrap(torch.cuda, 'synchronize', 'torch.cuda.synchronize')

    if launches:
        try:
            import warp as wp

            def wrapLaunch(attr):
                orig = getattr(wp, attr, None)
                if orig is None:
                    return

                def counted(kernel, *a, **k):
                    _record('launch', getattr(kernel, 'key', str(kernel)))
                    return orig(kernel, *a, **k)
                _patched.append((wp, attr, orig))
                setattr(wp, attr, counted)
            wrapLaunch('launch')
            wrapLaunch('launch_tiled')
            for attr in ('synchronize', 'synchronize_device'):
                orig = getattr(wp, attr, None)
                if orig is not None:
                    def counted(*a, _o=orig, _l=f'wp.{attr}', **k):
                        _record('sync', _l, _callSite() + f' [{_l}]')
                        return _o(*a, **k)
                    _patched.append((wp, attr, orig))
                    setattr(wp, attr, counted)
        except ImportError:
            pass
        graph = getattr(torch.cuda, 'CUDAGraph', None)
        if graph is not None:
            origReplay = graph.replay

            def replay(self, *a, **k):
                _record('replay', 'graph')
                return origReplay(self, *a, **k)
            _patched.append((graph, 'replay', origReplay))
            graph.replay = replay
    enabled = True


def disable():
    """Remove every hook (counts are kept; `reset()` clears them)."""
    global enabled, _dispatchMode
    if not enabled:
        return
    enabled = False
    if _dispatchMode is not None:
        _dispatchMode.__exit__(None, None, None)
        _dispatchMode = None
    for obj, attr, orig in reversed(_patched):
        setattr(obj, attr, orig)
    _patched.clear()
    if _origShowWarning is not None:
        warnings.showwarning = _origShowWarning
        try:
            import torch
            torch.cuda.set_sync_debug_mode('default')
        except Exception:                               # noqa: BLE001
            pass


# -- reporting ---------------------------------------------------------------------

def _per(n, calls):
    return n / calls if calls else 0.0


def report(top=20, detail=3, minOps=1):
    """Table of the sections by total ops: calls, ops / launches / syncs per call
    (inclusive, and `self` = excluding nested sections), CUDA-graph replays, and under
    each of the heaviest sections the most frequent torch ops, kernels and sync sites."""
    with _lock:
        rows = [(n, a) for n, a in _sections.items() if a.ops + a.launches + a.syncs >= minOps]
        rows.sort(key=lambda r: -(r[1].ops + 50 * r[1].launches + 1000 * r[1].syncs))
        steps = max(_stepIndex, 1)
        out = [f"call audit: {sum(a.calls for _, a in rows)} section entries over {_stepIndex} marked step(s)",
               f"{'section':<46} {'calls':>7} {'ops/call':>9} {'self':>8} {'launch/call':>11} {'sync/call':>9} "
               f"{'sync/step':>9} {'replays':>7}"]
        for n, a in rows[:top]:
            out.append(f"{n[:46]:<46} {a.calls:>7} {_per(a.ops, a.calls):>9.1f} {_per(a.selfOps, a.calls):>8.1f} "
                       f"{_per(a.launches, a.calls):>11.1f} {_per(a.syncs, a.calls):>9.2f} "
                       f"{a.syncs / steps:>9.2f} {a.replays:>7}")
        for n, a in rows[:detail]:
            out.append(f"  {n}")
            if a.opNames:
                out.append("    top ops     : " + ', '.join(f"{k} x{v / a.calls:.1f}" for k, v in a.opNames.most_common(6)))
            if a.kernelNames:
                out.append("    top kernels : " + ', '.join(f"{k} x{v / a.calls:.1f}" for k, v in a.kernelNames.most_common(5)))
            if a.syncSites:
                out.append("    sync sites  : " + ', '.join(f"{k} x{v}" for k, v in a.syncSites.most_common(5)))
        if _outside.ops or _outside.launches or _outside.syncs:
            out.append(f"{'(outside any section)':<46} {'':>7} ops {_outside.ops}, launches {_outside.launches}, "
                       f"syncs {_outside.syncs}")
            if _outside.syncSites:
                out.append("    sync sites  : " + ', '.join(f"{k} x{v}" for k, v in _outside.syncSites.most_common(5)))
        return '\n'.join(out)


def treeReport(minEvents=1):
    """Every section as a tree (each under its most frequent parent), with the same columns as
    `report`: inclusive ops / launches / syncs per call, `self` ops (excluding children) and
    syncs per step. The last lines say how much of the run happened outside any section."""
    with _lock:
        steps = max(_stepIndex, 1)
        primary = {n: (a.parents.most_common(1)[0][0] if a.parents else None) for n, a in _sections.items()}
        kids = collections.defaultdict(list)
        for n in _sections:
            kids[primary[n]].append(n)
        weight = lambda n: -(_sections[n].ops + 50 * _sections[n].launches + 1000 * _sections[n].syncs)
        out = [f"{'section (tree, inclusive unless "self")':<72} {'calls':>6} {'ops/call':>9} {'self ops':>9} "
               f"{'launch/call':>11} {'sync/call':>9} {'self sync':>9} {'sync/step':>9}"]

        def emit(n, depth):
            a = _sections[n]
            if a.ops + a.launches + a.syncs >= minEvents:
                label = ('  ' * depth + n)[:72]
                out.append(f"{label:<72} {a.calls:>6} {_per(a.ops, a.calls):>9.1f} {_per(a.selfOps, a.calls):>9.1f} "
                           f"{_per(a.launches, a.calls):>11.1f} {_per(a.syncs, a.calls):>9.2f} "
                           f"{_per(a.selfSyncs, a.calls):>9.2f} {a.syncs / steps:>9.2f}")
            for c in sorted(kids.get(n, ()), key=weight):
                emit(c, depth + 1)
        for root in sorted(kids.get(None, ()), key=weight):
            emit(root, 0)
        tot = _total
        pct = lambda part, whole: 100.0 * part / whole if whole else 100.0
        out.append(f"coverage: {pct(tot.ops - _outside.ops, tot.ops):.1f} % of {tot.ops} ops, "
                   f"{pct(tot.launches - _outside.launches, tot.launches):.1f} % of {tot.launches} launches, "
                   f"{pct(tot.syncs - _outside.syncs, tot.syncs):.1f} % of {tot.syncs} syncs inside a section "
                   f"({_stepIndex} marked steps)")
        if _outside.ops or _outside.launches or _outside.syncs:
            out.append(f"  OUTSIDE any section: ops {_outside.ops}, launches {_outside.launches}, syncs {_outside.syncs}")
            if _outside.syncSites:
                out.append("    sync sites  : " + ', '.join(f"{k} x{v}" for k, v in _outside.syncSites.most_common(6)))
            if _outside.opNames:
                out.append("    top ops     : " + ', '.join(f"{k} x{v}" for k, v in _outside.opNames.most_common(6)))
        return '\n'.join(out)


def stepReport(n=5):
    """The `n` marked steps with the most syncs (ties: most ops) and, for each, the
    sections that issued them: which steps spike and where."""
    with _lock:
        def key(rec):
            tot = rec[1]
            return (-sum(v[2] for v in tot.values()), -sum(v[0] for v in tot.values()))
        recs = sorted(_stepHistory, key=key)[:n]
        out = [f"worst {len(recs)} of the last {len(_stepHistory)} marked steps (by syncs, then ops):"]
        for idx, tot in recs:
            ops = max((v[0] for v in tot.values()), default=0)
            syncs = max((v[2] for v in tot.values()), default=0)
            out.append(f"  step {idx}: up to {ops} ops / {syncs} syncs in one section")
            for name, (o, l, s) in sorted(tot.items(), key=lambda kv: -(kv[1][2] * 1000 + kv[1][0]))[:4]:
                out.append(f"      {name[:50]:<50} ops {o:>6}  launches {l:>5}  syncs {s:>4}")
        return '\n'.join(out)


def printReport(file=None):
    file = file or sys.stderr
    print(treeReport(), file=file)
    print(report(top=12, detail=6), file=file)
    print(stepReport(), file=file)


if os.environ.get('WARPSPHCORE_AUDIT', '0') == '1':
    atexit.register(lambda: enabled and printReport())
    # torch and warp are imported lazily: the first record_function() call after they are
    # importable installs the hooks (see `profiling.record_function`)
    _autoEnable = True
else:
    _autoEnable = False
