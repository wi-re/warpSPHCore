"""Exact ideal-gas vacuum-opening fans shared by meshless Warp frontends.

A zero-pressure star region is physical when the two rarefaction tails separate;
retrying with a zero velocity jump introduces an artificial pressure force.
"""
import warp as wp
from typing import Any
from ..type_config import scalar_t

@wp.func
def opensVacuum(rhoL:scalar_t,uL:scalar_t,pL:scalar_t,rhoR:scalar_t,uR:scalar_t,pR:scalar_t,gamma:scalar_t,gammaRight:scalar_t):
    aL=wp.sqrt(gamma*pL/rhoL)
    aR=wp.sqrt(gammaRight*pR/rhoR)
    return uR-uL >= scalar_t(2.)*aL/(gamma-scalar_t(1.))+scalar_t(2.)*aR/(gammaRight-scalar_t(1.))

@wp.func
def vacuumContact(rhoL:scalar_t,uL:scalar_t,pL:scalar_t,rhoR:scalar_t,uR:scalar_t,pR:scalar_t,gamma:scalar_t,gammaRight:scalar_t):
    left=uL+scalar_t(2.)*wp.sqrt(gamma*pL/rhoL)/(gamma-scalar_t(1.))
    right=uR-scalar_t(2.)*wp.sqrt(gammaRight*pR/rhoR)/(gammaRight-scalar_t(1.))
    return scalar_t(.5)*left+scalar_t(.5)*right

@wp.func
def vacuumFaceFlux(rhoL:scalar_t,uL:scalar_t,vtL:Any,pL:scalar_t,rhoR:scalar_t,uR:scalar_t,vtR:Any,pR:scalar_t,
                   gamma:scalar_t,mode:wp.int32,normal:Any,frame:Any,gammaRight:scalar_t):
    contact=vacuumContact(rhoL,uL,pL,rhoR,uR,pR,gamma,gammaRight)
    zero=scalar_t(0.)
    if mode==0:
        return zero,vtL*zero,zero,contact,zero
    aL=wp.sqrt(gamma*pL/rhoL)
    aR=wp.sqrt(gammaRight*pR/rhoR)
    rho=zero
    u=zero
    pressure=zero
    tangential=vtL
    g=gamma
    if uL-aL >= zero:
        rho=rhoL
        u=uL
        pressure=pL
    elif uL+scalar_t(2.)*aL/(gamma-scalar_t(1.)) > zero:
        sound=(scalar_t(2.)*aL+(gamma-scalar_t(1.))*uL)/(gamma+scalar_t(1.))
        u=sound
        rho=rhoL*wp.pow(sound/aL,scalar_t(2.)/(gamma-scalar_t(1.)))
        pressure=pL*wp.pow(sound/aL,scalar_t(2.)*gamma/(gamma-scalar_t(1.)))
    elif uR+aR <= zero:
        rho=rhoR
        u=uR
        pressure=pR
        tangential=vtR
        g=gammaRight
    elif uR-scalar_t(2.)*aR/(gammaRight-scalar_t(1.)) < zero:
        sound=(scalar_t(2.)*aR-(gammaRight-scalar_t(1.))*uR)/(gammaRight+scalar_t(1.))
        u=-sound
        rho=rhoR*wp.pow(sound/aR,scalar_t(2.)/(gammaRight-scalar_t(1.)))
        pressure=pR*wp.pow(sound/aR,scalar_t(2.)*gammaRight/(gammaRight-scalar_t(1.)))
        tangential=vtR
        g=gammaRight
    mass=rho*u
    momentum=(mass*u+pressure)*normal+mass*tangential
    energy=(pressure/(g-scalar_t(1.))+scalar_t(.5)*rho*(u*u+wp.dot(tangential,tangential))+pressure)*u
    return mass,momentum+mass*frame,energy+wp.dot(frame,momentum)+scalar_t(.5)*wp.dot(frame,frame)*mass,contact,zero
