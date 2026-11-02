"""Small implicit solve using native Jv and J^T v, without a recorded solve graph.

This is a capability example, not a replacement for the repository's solvers.
The nonlinear residual is R(u) = u + dt*(u^3 - P u) - theta*source, where P is
normalized SPH interpolation on fixed geometry. Dense matrices are formed only
after the matrix-free solves, as an independent correctness check.
"""
import argparse
import json
from pathlib import Path
import torch

from template_runtime import Library,Plan,transpose
from verify import geometry,displacement


def gmres(action,rhs,tolerance=2e-12,maximum=48):
    """Unrestarted modified-Gram-Schmidt GMRES for this small example."""
    beta=torch.linalg.vector_norm(rhs)
    if float(beta)==0:return torch.zeros_like(rhs),0
    basis=[rhs/beta]
    hessenberg=rhs.new_zeros((maximum+1,maximum))
    target=rhs.new_zeros(maximum+1);target[0]=beta
    for column in range(maximum):
        candidate=action(basis[column])
        # Reorthogonalize: the dense check should measure the operator, rather
        # than loss of orthogonality in this deliberately compact GMRES code.
        for _ in range(2):
            for row,vector in enumerate(basis):
                coefficient=torch.dot(vector,candidate)
                hessenberg[row,column]+=coefficient
                candidate-=coefficient*vector
        length=torch.linalg.vector_norm(candidate)
        hessenberg[column+1,column]=length
        projected=hessenberg[:column+2,:column+1]
        coefficients=torch.linalg.lstsq(projected,target[:column+2]).solution
        solution=sum(coefficient*vector for coefficient,vector in zip(coefficients,basis))
        residual=torch.linalg.vector_norm(action(solution)-rhs)
        if float(residual)<=tolerance*max(1.,float(beta)):
            return solution,column+1
        if float(length)<1e-15:break
        basis.append(candidate/length)
    raise RuntimeError(f'Example GMRES did not converge: residual={float(residual)}')


def run(library,device):
    n=48;dtype=torch.float64
    positions=(torch.arange(n,device=device,dtype=dtype)+.5)[:,None]/n
    data=dict(qx=positions,rx=positions,h=positions.new_full((n,),4/n),
        mass=positions.new_full((n,),1/n),density=positions.new_ones(n),qvalue=None,
        lower=positions.new_zeros(1),upper=positions.new_ones(1),
        periodic=torch.ones(1,device=device,dtype=torch.bool))
    distance=torch.linalg.vector_norm(displacement(data),dim=-1)
    membership=distance<data['h'][:,None]
    _,indices=torch.where(membership);counts=membership.sum(1).int()
    offsets=torch.cat([counts.new_zeros(1,dtype=torch.int64),counts.long().cumsum(0)])[:-1]
    forward=Plan(dict(indices=indices.int(),counts=counts,offsets=offsets))
    reverse=transpose(forward,n,n)
    source=torch.sin(2*torch.pi*positions[:,0])
    theta=.2;dt=.15

    def fields(value):return dict(data,rvalue=value[:,None].contiguous())
    def interpolate(value,save=False):return library.forward(fields(value),forward,5,save=save)
    def residual(value):return value+dt*(value**3-interpolate(value).values[:,0])-theta*source

    value=torch.zeros_like(source);iterations=[]
    for newton in range(8):
        current=residual(value)
        if float(current.norm())<2e-12:break
        # Geometry, mass and density have no tangents in this implicit unknown.
        # The native dispatch therefore selects its field-only specialization.
        def jvp(direction):
            tangents=(None,)*6+(direction[:,None].contiguous(),)
            product=library.derivative(fields(value),forward,None,5,tangents=tangents)[:,0]
            return direction+dt*(3*value.square()*direction-product)
        update,count=gmres(jvp,-current)
        value=value+update;iterations.append(count)
    else:raise RuntimeError('Example Newton solve did not converge')

    # Implicit backward uses the converged state, not a tape of Newton or GMRES
    # iterations. For L=0.5*||u||^2, solve R_u^T lambda = u.
    saved=interpolate(value,save=True)
    reference_gradient=torch.empty((n,1),device=device,dtype=dtype)
    def transpose_product(direction):
        requested=(None,)*6+(reference_gradient,)
        gradients=library.derivative(fields(value),forward,reverse,5,
            cotangent=direction[:,None].contiguous(),cache=saved.cache,gradients=requested)
        return direction+dt*(3*value.square()*direction-gradients[6][:,0])
    multiplier,transpose_iterations=gmres(transpose_product,value)
    parameter_gradient=torch.dot(multiplier,source)

    # Independent dense geometry and linear algebra are verification only.
    _,kernel,_,_=geometry(dict(data,rvalue=value[:,None]))
    weights=kernel*data['mass']/data['density']
    interpolation=weights/weights.sum(1,keepdim=True)
    jacobian=torch.eye(n,device=device,dtype=dtype)+dt*(torch.diag(3*value.square())-interpolation)
    expected=torch.linalg.solve(jacobian.T,value)
    probe=torch.cos(torch.arange(n,device=device,dtype=dtype))
    jvp_error=float((jvp(probe)-jacobian@probe).norm())
    vjp_error=float((transpose_product(probe)-jacobian.T@probe).norm())
    adjoint_error=float((multiplier-expected).norm())
    gradient_error=float(torch.abs(parameter_gradient-torch.dot(expected,source)))
    assert float(residual(value).norm())<2e-12
    assert max(jvp_error,vjp_error,adjoint_error,gradient_error)<2e-10
    return dict(device=device,newton_iterations=len(iterations),gmres_iterations=iterations,
        transpose_gmres_iterations=transpose_iterations,residual_norm=float(residual(value).norm()),
        jvp_dense_error=jvp_error,vjp_dense_error=vjp_error,adjoint_solve_error=adjoint_error,
        parameter_gradient=float(parameter_gradient),parameter_gradient_error=gradient_error)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True);parser.add_argument('--gpu',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(24)
    with torch.no_grad():
        library=Library(args.cpu,args.gpu)
        result=[run(library,'cpu')]
        if args.gpu:result.append(run(library,'cuda'));torch.cuda.synchronize()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
