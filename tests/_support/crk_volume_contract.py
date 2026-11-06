"""CRK reciprocal-volume fixtures, independent sum equation and backend entry."""
from dataclasses import replace
import torch
from .density_contract import case as density_case,oracle as density_equation


def case(*args,**kwargs):
    q,r,d,p,g,v,states,_=density_case(*args,**kwargs)
    def functions(backend,properties=p,adjacency=g):
        if backend=='native':from warpSPHCore.crk.native.nt_volume import computeCRKVolume as compute
        else:from warpSPHCore.crk.crk_volume import _computeCRKVolume_stateBackend as compute
        def run(*values):
            qi,rj,domain=states(values)
            return compute(qi,properties,domain,adjacency,rj)
        return run
    return q,r,d,p,g,v,states,functions


def oracle(values,domain,properties,graph):
    unit=list(values);unit[4]=torch.ones_like(values[4])
    total=density_equation(unit,domain,properties,graph)
    safe=torch.where(total!=0,total,torch.ones_like(total))
    return torch.where(total!=0,1/safe,torch.zeros_like(total))
