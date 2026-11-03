"""Small execution boundary shared by the inference workspace controllers."""
from ..._backend import BACKEND

if BACKEND == 'native':
    from ..native.nt_krylov import engine
else:
    from contextlib import contextmanager
    from importlib import import_module
    import warp as wp
    from ...autograd.stream import torchWarpStream, torchWarpCapture

    class WarpExecution:
        initialize = staticmethod(wp.init)
        array = staticmethod(wp.from_torch)
        launch = staticmethod(wp.launch)

        def scalar(self,value,dtype):
            return wp.dtype_from_torch(dtype)(value)

        def inner(self,left,right,*,out):
            return wp.utils.array_inner(left,right,out=out)

        def kernels(self,family):
            return import_module('.wp_'+family,__package__)

        @contextmanager
        def scope(self,device):
            with wp.ScopedDevice(wp.device_from_torch(device)),torchWarpStream(device):
                yield

        def capture(self,graph):
            return torchWarpCapture(graph)

    engine=WarpExecution()
