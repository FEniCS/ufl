"""This module extends the form language with free function operators,
which are either already available as member functions on UFL objects
or defined as compound operators involving basic operations on the UFL
objects."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-04-09 -- 2008-08-18"

# Modified by Anders Logg, 2008

from .differentiation import Diff, Grad, Div, Curl, Rot
from .tensoralgebra import Transposed, Inner, Outer, Dot, Cross, Determinant, Inverse, Trace, Deviatoric, Cofactor
from .variable import Variable
from .restriction import PositiveRestricted, NegativeRestricted
from .geometry import FacetNormal

#--- Tensor operators ---

def transpose(v):
    "Return transpose of expression"
    return Transposed(v)

def outer(v, w):
    return Outer(v, w)

def inner(v, w):
    return Inner(v, w)

def dot(v, w):
    return Dot(v, w)

def cross(v, w):
    return Cross(v, w)

def det(v):
    return Determinant(v)

def inv(v):
    return Inverse(v)

def tr(v):
    return Trace(v)

def dev(v):
    return Deviatoric(v)

def cofac(v):
    return Cofactor(v)

#--- Differential operators

# FIXME: Do we want this? Isn't f.dx(i) enough?
# It is good to have when differentiating a large expression which looks silly
# when appending .dx():   (v*u*....).dx(i)

def Dx(v, i): 
    "Return the partial derivative with respect to spatial variable number i."
    return v.dx(i)

def Dt(v): # FIXME: Add class
    #return TimeDerivative(v)
    raise NotImplementedError

# FIXME: What is Diff?
def diff(v, x):
    return Diff(v, x)

def grad(v):
    return Grad(v)

def div(v):
    return Div(v)

def curl(v):
    return Curl(v)

def rot(v):
    return Rot(v)

#--- DG operators ---

def jump(v):
    if v.rank() == 0:
        n = FacetNormal()
        return v('+')*n('+') + v('-')*n('-')
    elif v.rank() == 1:
        n = FacetNormal()
        return dot(v('+'), n('+')) + dot(v('-'), n('-'))
    else:
        ufl_error("jump(v) is only defined when v scalar or a vector (not rank %d) expressions." % v.rank())

def avg(v):
    return 0.5*(v('+') + v('-'))

#--- Other operators ---

def variable(v):
    return Variable(v)
