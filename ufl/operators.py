"""This module extends the form language with free function operators,
which are either already available as member functions on UFL objects
or defined as compound operators involving basic operations on the UFL
objects."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-04-09 -- 2008-05-20"

from differentiation import Grad, Div, Curl, Rot
from tensoralgebra import Transpose, Inner, Outer, Dot, Cross, Determinant, Inverse, Trace, Deviatoric, Cofactor

#--- Tensor operators ---

def transpose(o):
    "Return transpose of expression"
    return Transpose(o)

def outer(a, b):
    return Outer(a, b)

def inner(a, b):
    return Inner(a, b)

def dot(a, b):
    return Dot(a, b)

def cross(a, b):
    return Cross(a, b)

def det(f):
    return Determinant(f)

def inv(f):
    return Inverse(f)

def tr(f):
    return Trace(f)

def dev(A):
    return Deviatoric(A)

def cofac(A):
    return Cofactor(A)

#--- Differential operators

def Dx(f, i): # FIXME: Do we want this? Isn't f.dx(i) enough?
    "Return the partial derivative with respect to spatial variable number i."
    return f.dx(i)

def Dt(o): # FIXME: Add class
    #return TimeDerivative(o)
    raise NotImplementedError

def grad(f):
    return Grad(f)

def div(f):
    return Div(f)

def curl(f):
    return Curl(f)

def rot(f):
    return Rot(f)

#--- DG operators ---

def jump(o):
    raise NotImplementedError

def avg(o):
    raise NotImplementedError

#--- Suggestions --- # TODO: This is already in mathfunctions?

def sqrt(o):
    raise NotImplementedError

