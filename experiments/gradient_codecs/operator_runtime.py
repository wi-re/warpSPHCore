"""Borrowed-buffer primal wrappers for the heavier-operator codec experiment."""
import ctypes as C
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'csr_small_gaps'))
from gaps_runtime import Library as GapLibrary, Case, descriptor, build_plan, BAND, NIBBLE, RAW32, RAW64, DELTA
# gaps_runtime adds the byte experiment to sys.path. Load it under an explicit
# name because this module itself is also named runtime.
import importlib.util
_spec=importlib.util.spec_from_file_location('gradient_byte_runtime',Path(__file__).resolve().parents[1]/'csr_compression/runtime.py')
byte=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=byte
_spec.loader.exec_module(byte)

NAMES={RAW64:'csr64',RAW32:'csr32',DELTA:'delta',BAND:'band',NIBBLE:'nibble'}
WORKLOADS=('gradient_scalar','gradient_vector','divergence','covariance')


class GradientCase(C.Structure):
    _fields_=[('pairs',Case)]+[(name,C.c_void_p) for name in ('volumes','A','B','gradA','gradB')]


class Library(GapLibrary):
    def __init__(self,path):
        super().__init__(path)
        self.library.gradient_consume.argtypes=[C.POINTER(GradientCase)]+[C.c_int]*9
        self.library.gradient_consume.restype=C.c_int

    def run(self,request,encoding,work,corrected,chunk,items,batch,threads=128,pack=True,iterations=1):
        request.pairs.stream=torch.cuda.current_stream().cuda_stream
        self.check(self.library.gradient_consume(C.byref(request),encoding,work,corrected,
                                                chunk,items,batch,threads,pack,iterations))


def request_for(common,topology,n,dimension,precision,volumes=None,correction=None):
    request=GradientCase(descriptor(dict(common,**topology),n,dimension=dimension,precision=precision))
    if volumes is not None: request.volumes=volumes.data_ptr()
    if correction is not None:
        for name in ('A','B','gradA','gradB'):setattr(request,name,getattr(correction,name).data_ptr())
    return request


def field_count(work,dimension):
    return 1 if work==0 else 0 if work==3 else dimension


def output_shape(work,n,dimension):
    return (n,dimension) if work==0 else (n,) if work==2 else (n,dimension,dimension)


def packing_pitch(work,dimension):
    return ((dimension+1+field_count(work,dimension)+3)//4)*4


def encode(library,indices,counts,offsets,chunk,encoding):
    # Call the byte ABI through its own wrapper. No CSR pointer is retained by
    # any compressed consumer; the input graph is only an encoder dependency.
    if encoding==DELTA:
        return byte.build_plan(byte.Library(library.library._name),indices,counts,offsets,chunk,encoding)
    return build_plan(library,indices,counts,offsets,chunk,encoding)
