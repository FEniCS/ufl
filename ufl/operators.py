"""This module extends the form language with free function operators,
which are either already available as member functions on UFL objects
or defined as compound operators involving basic operations on the UFL
objects."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-04-09 -- 2008-08-20"

from .differentiation import Diff, Grad, Div, Curl, Rot
from .tensoralgebra import Transposed, Inner, Outer, Dot, Cross, Determinant, Inverse, Trace, Deviatoric, Cofactor
from .variable import Variable
from .conditional import EQ, NE, LE, GE, LT, GT, Conditional
from .mathfunctions import Sqrt, Exp, Ln, Cos, Sin

#--- Tensor operators ---

def transpose(o):
    "Return transpose of expression"
    return Transposed(o)

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

def diff(f, x):
    return Diff(f, x)

def grad(f):
    return Grad(f)

def div(f):
    return Div(f)

def curl(f):
    return Curl(f)

def rot(f):
    return Rot(f)

#--- DG operators ---

def jump(o): # FIXME
    raise NotImplementedError 

def avg(o): # FIXME
    raise NotImplementedError

#--- Other operators ---

def variable(o):
    return Variable(o)

#--- Conditional expressions ---

def conditional(condition, true_value, false_value):
    "A conditional expression, like the C construct (condition ? true_value : false_value)."
    return Conditional(condition, true_value, false_value)

def eq(left, right):
    "A boolean expresion (left == right) for use with conditional."
    return EQ(left, right)

def ne(left, right):
    "A boolean expresion (left != right) for use with conditional."
    return NE(left, right)

def le(left, right):
    "A boolean expresion (left <= right) for use with conditional."
    return LE(left, right)

def ge(left, right):
    "A boolean expresion (left >= right) for use with conditional."
    return GE(left, right)

def lt(left, right):
    "A boolean expresion (left < right) for use with conditional."
    return LT(left, right)

def gt(left, right):
    "A boolean expresion (left > right) for use with conditional."
    return GT(left, right)

#--- Math functions ---

def sqrt(f):
    return Sqrt(f)

def exp(f):
    return Exp(f)

def ln(f):
    return Ln(f)

def cos(f):
    return Cos(f)

def sin(f):
    return Sin(f)

