"""Borrowed-buffer descriptors for the private tuning experiment."""
import ctypes as C

class CsrCase(C.Structure):
    _fields_ = [(name, C.c_int) for name in ['dimension', 'precision', 'particles', 'index_bits']]
    _fields_ += [(name, C.c_void_p) for name in ['stream', 'positions', 'supports', 'masses', 'densities',
        'values', 'offsets', 'indices', 'counts', 'lower', 'upper', 'periodic', 'packed', 'output']]
