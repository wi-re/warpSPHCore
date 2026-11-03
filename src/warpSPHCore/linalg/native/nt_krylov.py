"""Native pointwise workspace updates; solve control flow stays shared."""
from contextlib import nullcontext, contextmanager
import gc
from types import SimpleNamespace
import torch

# Task IDs mirror csrc_krylov/nt_krylov_rows.h. Only the module adapter sees
# this layout; solver callers continue using CGWorkspace/BiCGStabWorkspace.
_TASKS = {
    0: (7, ('history',)),  # cg_initialize
    1: (9, ('threshold', 'hasThreshold', 'dim')),  # cg_candidate
    2: (8, ('history',)),  # cg_decide
    3: (5, ()),  # cg_commit
    4: (3, ()),  # cg_diagonal
    5: (5, ()),  # cg_direction
    6: (6, ()),  # bicgstab_initialize
    7: (7, ()),  # bicgstab_alpha
    8: (8, ('threshold', 'hasThreshold', 'dim')),  # bicgstab_halfCandidate
    9: (4, ()),  # bicgstab_halfDecision
    10: (3, ()),  # bicgstab_omega
    11: (10, ('threshold', 'hasThreshold', 'dim')),  # bicgstab_fullCandidate
    12: (10, ()),  # bicgstab_fullDecision
    13: (10, ()),  # bicgstab_direction
    14: (6, ('history',)),  # bicgstab_finish
    15: (5, ()),  # bicgstab_commit
    16: (3, ()),  # bicgstab_diagonal
}
_KERNELS = {
    'cg': SimpleNamespace(_initialize=0, _candidate=1, _decide=2, _commit=3, _diagonal=4, _direction=5),
    'bicgstab': SimpleNamespace(_initialize=6, _alpha=7, _halfCandidate=8, _halfDecision=9, _omega=10, _fullCandidate=11, _fullDecision=12, _direction=13, _finish=14, _commit=15, _diagonal=16),
}


class Command:
    def __init__(self,task,size,inputs):
        from ...native._build import build
        count,parameters=_TASKS[task]
        options=dict(threshold=0.,hasThreshold=False,dim=1,history=False)
        options.update(zip(parameters,inputs[count:],strict=True))
        self.operator=build().krylov_update
        self.arguments=(list(inputs[:count]),task,size,float(options['threshold']),
                        bool(options['hasThreshold']),int(options['dim']),bool(options['history']))

    def launch(self):
        self.operator(*self.arguments)


class NativeExecution:
    @staticmethod
    def initialize():
        pass

    @staticmethod
    def array(value):
        return value

    @staticmethod
    def inner(left,right,*,out):
        torch.dot(left,right,out=out.view(()))

    @staticmethod
    def kernels(family):
        return _KERNELS[family]

    @staticmethod
    def scope(device):
        return torch.cuda.device(device) if torch.device(device).type=='cuda' else nullcontext()

    @staticmethod
    @contextmanager
    def capture(graph):
        # The workspace has already warmed and synchronized its setup stream.
        # Capture that stream without torch.cuda.graph's device-wide sync.
        collect = gc.isenabled()
        gc.disable()
        try:
            graph.capture_begin()
            try:
                yield
            finally:
                graph.capture_end()
        finally:
            if collect:
                gc.enable()

    @staticmethod
    def scalar(value,dtype):
        return float(value)

    @staticmethod
    def launch(task,*,dim,inputs,record_cmd=False):
        command=Command(task,dim,inputs)
        if record_cmd:return command
        command.launch()


engine=NativeExecution()
