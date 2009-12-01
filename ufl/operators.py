"""This module extends the form language with free function operators,
which are either already available as member functions on UFL objects
or defined as compound operators involving basic operations on the UFL
objects."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-04-09 -- 2009-04-07"

import math
from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.constantvalue import Zero, ScalarValue, as_ufl
from ufl.differentiation import VariableDerivative, Grad, Div, Curl
from ufl.tensoralgebra import Transposed, Inner, Outer, Dot, Cross, Determinant, Inverse, Cofactor, Trace, Deviatoric, Skew, Sym
from ufl.variable import Variable
from ufl.tensors import as_tensor
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ufl.mathfunctions import Sqrt, Exp, Ln, Cos, Sin
from ufl.indexing import indices
from ufl.geometry import SpatialCoordinate

#--- Basic operators ---

def rank(f):
    "The rank of f."
    return len(f.shape())

def shape(f):
    "The shape of f."
    return f.shape()

#--- Tensor operators ---

def transpose(A):
    "The transposed of A."
    if A.shape() == ():
        return A
    return Transposed(A)

def outer(a, b):
    "The outer product of a and b."
    if a.shape() == () and b.shape() == ():
        return a*b
    return Outer(a, b)

def inner(a, b):
    "The inner product of a and b."
    if a.shape() == () and b.shape() == ():
        return a*b
    return Inner(a, b)
    #return contraction(a, range(a.rank()), b, range(b.rank()))

def dot(a, b):
    "The dot product of a and b."
    if a.shape() == () and b.shape() == ():
        return a*b
    return Dot(a, b)
    #return contraction(a, (a.rank()-1,), b, (b.rank()-1,))

def contraction(a, ai, b, bi):
    "The contraction of a and b over given axes."
    ufl_assert(len(ai) == len(bi), "Contraction must be over the same number of axes.")
    ash = a.shape()
    bsh = b.shape()
    aii = indices(a.rank())
    bii = indices(b.rank())
    cii = indices(len(ai))
    shape = [None]*len(ai)
    for i,j in enumerate(ai):
        aii[j] = cii[i]
        shape[i] = ash[j]
    for i,j in enumerate(bi):
        bii[j] = cii[i]
        ufl_assert(shape[i] == bsh[j], "Shape mismatch in contraction.")
    s = a[aii]*b[bii]
    cii = set(cii)
    ii = tuple(i for i in (aii + bii) if not i in cii)
    return as_tensor(s, ii)

def cross(a, b):
    "The cross product of a and b."
    return Cross(a, b)

def det(A):
    "The determinant of A."
    if A.shape() == ():
        return A
    return Determinant(A)

def inv(A):
    "The inverse of A."
    if A.shape() == ():
        return 1/A
    return Inverse(A)

def cofac(A):
    "The cofactor of A."
    return Cofactor(A)

def tr(A):
    "The trace of A."
    return Trace(A)

def dev(A):
    "The deviatoric part of A."
    return Deviatoric(A)

def skew(A):
    "The skew symmetric part of A."
    return Skew(A)

def sym(A):
    "The symmetric part of A."
    return Sym(A)

#--- Differential operators

def Dx(f, *i):
    "The partial derivative of f with respect to spatial variable number i. Equivalent to f.dx(*i)."
    return f.dx(*i)

def Dt(f):
    #return TimeDerivative(f) # TODO: Add class
    raise NotImplementedError

def Dn(f):
    "The directional derivative of f in the facet normal direction, Dn(f) := dot(n, grad(f))."
    cell = f.cell()
    if cell is None:
        return Zero(f.shape(), f.free_indices(), f.index_dimensions())
    return dot(grad(f), cell.n)

# TODO: We have "derivative", "diff", "Dx", and "f.dx(i)", can we unify these with more intuitive consistent naming?
def diff(f, v):
    """The derivative of f with respect to the variable v.

    If f is a form, diff is applied to each integrand.
    """
    if isinstance(f, Form):
        from ufl.algorithms.transformations import transform_integrands
        def _diff(e):
            return diff(e, v)
        return transform_integrands(f, _diff)

    if isinstance(v, SpatialCoordinate):
        r = v.rank()
        ii = indices(r + 1)
        if r:
            v = v[ii[:-1]]
        dv = v.dx(ii[-1])

    return VariableDerivative(f, v)

def diff2(f, v): # DIFFSHAPE TODO: Replace diff with this? The difference is that the shape of v comes before the shape of f in this version.
    "The derivative of f with respect to the variable v."
    if isinstance(f, Form):
        from ufl.algorithms.transformations import transform_integrands
        def _diff(e):
            return diff(e, v)
        return transform_integrands(f, _diff)

    if isinstance(v, SpatialCoordinate):
        r = v.rank()
        ii = indices(r + 1)
        if r:
            v = v[ii[1:]]
        dv = v.dx(ii[0])
        return as_tensor(dv, ii)
    return VariableDerivative(f, v)

def grad(f):
    "The gradient of f."
    return Grad(f)

def div(f):
    "The divergence of f."
    return Div(f)

def curl(f):
    "The curl of f."
    return Curl(f)
rot = curl

#--- DG operators ---

def jump(v, n=None):
    "The jump of v across a facet."
    cell = v.cell()
    if cell is None:
        warning("TODO: Not all expressions have a cell. Is it right to return zero from jump then?")
        # TODO: Is this right? If v has no cell, it doesn't depend on
        # anything spatially variable or any form arguments, and thus
        # the jump is zero. In other words, I'm assuming that
        # "v.cell() is None" is equivalent with "v is a constant".
        return Zero(v.shape(), v.free_indices(), v.index_dimensions())
    else:
        if n is None:
            return v('+') - v('-')
        r = v.rank()
        if r == 0:
            return v('+')*n('+') + v('-')*n('-')
        elif r == 1:
            return dot(v('+'), n('+')) + dot(v('-'), n('-'))

    error("jump(v, n) is only defined for scalar or vector-valued expressions (not rank %d expressions)." % r)

def avg(v):
    "The average of v across a facet."
    return 0.5*(v('+') + v('-'))

#--- Other operators ---

def variable(e):
    "A variable representing the given expression."
    return Variable(e)

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

def sign(x):
    "The sign (+1 or -1) of x."
    # TODO: Add a Sign type for this?
    return conditional(eq(x, 0), 0, conditional(lt(x, 0), -1, +1))

#--- Math functions ---

def _mathfunction(f, cls, fun):
    f = as_ufl(f)
    #if isinstance(f, ScalarValue):
    #    return as_ufl(fun(f._value))
    #if isinstance(f, Zero):
    #    return as_ufl(fun(0))
    r = cls(f)
    if isinstance(r, (ScalarValue, Zero, int, float)):
        return float(r)
    return r

def sqrt(f):
    "The square root of f."
    return _mathfunction(f, Sqrt, math.sqrt)

def exp(f):
    "The exponential of f."
    return _mathfunction(f, Exp, math.exp)

def ln(f):
    "The natural logarithm of f."
    return _mathfunction(f, Ln, math.log)

def cos(f):
    "The cosinus of f."
    return _mathfunction(f, Cos, math.cos)

def sin(f):
    "The sinus of f."
    return _mathfunction(f, Sin, math.sin)

