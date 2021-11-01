# -*- coding: utf-8 -*-
"""This module extends the form language with free function operators,
which are either already available as member functions on UFL objects
or defined as compound operators involving basic operations on the UFL
objects."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard, 2011
# Modified by Massimiliano Leoni, 2016.

import operator

from ufl.log import error, warning
from ufl.form import Form
from ufl.constantvalue import Zero, RealValue, ComplexValue, as_ufl
from ufl.differentiation import VariableDerivative, Grad, Div, Curl, NablaGrad, NablaDiv
from ufl.tensoralgebra import Transposed, Inner, Outer, Dot, Cross, \
    Determinant, Inverse, Cofactor, Trace, Deviatoric, Skew, Sym
from ufl.coefficient import Coefficient
from ufl.variable import Variable
from ufl.tensors import as_tensor, as_matrix, as_vector, ListTensor
from ufl.conditional import EQ, NE, \
    AndCondition, OrCondition, NotCondition, Conditional, MaxValue, MinValue
from ufl.algebra import Conj, Real, Imag
from ufl.mathfunctions import Sqrt, Exp, Ln, Erf,\
    Cos, Sin, Tan, Cosh, Sinh, Tanh, Acos, Asin, Atan, Atan2,\
    BesselJ, BesselY, BesselI, BesselK
from ufl.averaging import CellAvg, FacetAvg
from ufl.core.multiindex import indices
from ufl.indexed import Indexed
from ufl.geometry import SpatialCoordinate, FacetNormal
from ufl.checks import is_cellwise_constant
from ufl.domain import extract_domains
from ufl.core.external_operator import ExternalOperator

# --- Basic operators ---


def rank(f):
    "UFL operator: The rank of *f*."
    f = as_ufl(f)
    return len(f.ufl_shape)


def shape(f):
    "UFL operator: The shape of *f*."
    f = as_ufl(f)
    return f.ufl_shape


# --- Complex operators ---

def conj(f):
    "UFL operator: The complex conjugate of *f*"
    f = as_ufl(f)
    return Conj(f)


# Alias because both conj and conjugate are in numpy and we wish to be consistent.
conjugate = conj


def real(f):
    "UFL operator: The real part of *f*"
    f = as_ufl(f)
    return Real(f)


def imag(f):
    "UFL operator: The imaginary part of *f*"
    f = as_ufl(f)
    return Imag(f)


# --- Elementwise tensor operators ---

def elem_op_items(op_ind, indices, *args):
    sh = args[0].ufl_shape
    indices = tuple(indices)
    n = sh[len(indices)]

    def extind(ii):
        return indices + (ii,)

    if len(sh) == len(indices) + 1:
        return [op_ind(extind(i), *args) for i in range(n)]
    else:
        return [elem_op_items(op_ind, extind(i), *args) for i in range(n)]


def elem_op(op, *args):
    "UFL operator: Take the elementwise application of operator *op* on scalar values from one or more tensor arguments."
    args = [as_ufl(arg) for arg in args]
    sh = args[0].ufl_shape
    if not all(sh == x.ufl_shape for x in args):
        error("Cannot take elementwise operation with different shapes.")

    if sh == ():
        return op(*args)

    def op_ind(ind, *args):
        return op(*[x[ind] for x in args])
    return as_tensor(elem_op_items(op_ind, (), *args))


def elem_mult(A, B):
    "UFL operator: Take the elementwise multiplication of tensors *A* and *B* with the same shape."
    return elem_op(operator.mul, A, B)


def elem_div(A, B):
    "UFL operator: Take the elementwise division of tensors *A* and *B* with the same shape."
    return elem_op(operator.truediv, A, B)


def elem_pow(A, B):
    "UFL operator: Take the elementwise power of tensors *A* and *B* with the same shape."
    return elem_op(operator.pow, A, B)


# --- Tensor operators ---

def transpose(A):
    "UFL operator: Take the transposed of tensor A."
    A = as_ufl(A)
    if A.ufl_shape == ():
        return A
    return Transposed(A)


def outer(*operands):
    "UFL operator: Take the outer product of two or more operands. The complex conjugate of the first argument is taken."
    n = len(operands)
    if n == 1:
        return operands[0]
    elif n == 2:
        a, b = operands
    else:
        a = outer(*operands[:-1])
        b = operands[-1]
    a = as_ufl(a)
    b = as_ufl(b)
    if a.ufl_shape == () and b.ufl_shape == ():
        return Conj(a) * b
    return Outer(a, b)


def inner(a, b):
    "UFL operator: Take the inner product of *a* and *b*. The complex conjugate of the second argument is taken."
    a = as_ufl(a)
    b = as_ufl(b)
    if a.ufl_shape == () and b.ufl_shape == ():
        return a * Conj(b)
    return Inner(a, b)


# TODO: Something like this would be useful in some cases, but should
# inner just support len(a.ufl_shape) != len(b.ufl_shape) instead?
def _partial_inner(a, b):
    "UFL operator: Take the partial inner product of a and b."
    ar, br = len(a.ufl_shape), len(b.ufl_shape)
    n = min(ar, br)
    return contraction(a, list(range(n - ar, n - ar + n)), b, list(range(n)))


def dot(a, b):
    "UFL operator: Take the dot product of *a* and *b*. This won't take the complex conjugate of the second argument."
    a = as_ufl(a)
    b = as_ufl(b)
    if a.ufl_shape == () and b.ufl_shape == ():
        return a * b
    return Dot(a, b)


def contraction(a, a_axes, b, b_axes):
    "UFL operator: Take the contraction of a and b over given axes."
    ai, bi = a_axes, b_axes
    if len(ai) != len(bi):
        error("Contraction must be over the same number of axes.")
    ash = a.ufl_shape
    bsh = b.ufl_shape
    aii = indices(len(a.ufl_shape))
    bii = indices(len(b.ufl_shape))
    cii = indices(len(ai))
    shape = [None] * len(ai)
    for i, j in enumerate(ai):
        aii[j] = cii[i]
        shape[i] = ash[j]
    for i, j in enumerate(bi):
        bii[j] = cii[i]
        if shape[i] != bsh[j]:
            error("Shape mismatch in contraction.")
    s = a[aii] * b[bii]
    cii = set(cii)
    ii = tuple(i for i in (aii + bii) if i not in cii)
    return as_tensor(s, ii)


def perp(v):
    "UFL operator: Take the perp of *v*, i.e. :math:`(-v_1, +v_0)`."
    v = as_ufl(v)
    if v.ufl_shape != (2,):
        error("Expecting a 2D vector expression.")
    return as_vector((-v[1], v[0]))


def cross(a, b):
    "UFL operator: Take the cross product of *a* and *b*."
    a = as_ufl(a)
    b = as_ufl(b)
    return Cross(a, b)


def det(A):
    "UFL operator: Take the determinant of *A*."
    A = as_ufl(A)
    if A.ufl_shape == ():
        return A
    return Determinant(A)


def inv(A):
    "UFL operator: Take the inverse of *A*."
    A = as_ufl(A)
    if A.ufl_shape == ():
        return 1 / A
    return Inverse(A)


def cofac(A):
    "UFL operator: Take the cofactor of *A*."
    A = as_ufl(A)
    return Cofactor(A)


def tr(A):
    "UFL operator: Take the trace of *A*."
    A = as_ufl(A)
    return Trace(A)


def diag(A):
    """UFL operator: Take the diagonal part of rank 2 tensor *A* **or**
    make a diagonal rank 2 tensor from a rank 1 tensor.

    Always returns a rank 2 tensor. See also ``diag_vector``."""

    # TODO: Make a compound type or two for this operator

    # Get and check dimensions
    r = len(A.ufl_shape)
    if r == 1:
        n, = A.ufl_shape
    elif r == 2:
        m, n = A.ufl_shape
        if m != n:
            error("Can only take diagonal of square tensors.")
    else:
        error("Expecting rank 1 or 2 tensor.")

    # Build matrix row by row
    rows = []
    for i in range(n):
        row = [0] * n
        row[i] = A[i] if r == 1 else A[i, i]
        rows.append(row)
    return as_matrix(rows)


def diag_vector(A):
    """UFL operator: Take the diagonal part of rank 2 tensor *A* and return as a vector.

    See also ``diag``."""

    # TODO: Make a compound type for this operator

    # Get and check dimensions
    if len(A.ufl_shape) != 2:
        error("Expecting rank 2 tensor.")
    m, n = A.ufl_shape
    if m != n:
        error("Can only take diagonal of square tensors.")

    # Return diagonal vector
    return as_vector([A[i, i] for i in range(n)])


def dev(A):
    "UFL operator: Take the deviatoric part of *A*."
    A = as_ufl(A)
    return Deviatoric(A)


def skew(A):
    "UFL operator: Take the skew symmetric part of *A*."
    A = as_ufl(A)
    return Skew(A)


def sym(A):
    "UFL operator: Take the symmetric part of *A*."
    A = as_ufl(A)
    return Sym(A)


# --- Differential operators

def Dx(f, *i):
    """UFL operator: Take the partial derivative of *f* with respect
    to spatial variable number *i*. Equivalent to ``f.dx(*i)``."""
    f = as_ufl(f)
    return f.dx(*i)


def Dt(f):
    "UFL operator: <Not implemented yet!> The partial derivative of *f* with respect to time."
    raise NotImplementedError


def Dn(f):
    """UFL operator: Take the directional derivative of *f* in the
    facet normal direction, Dn(f) := dot(grad(f), n)."""
    f = as_ufl(f)
    if is_cellwise_constant(f):
        return Zero(f.ufl_shape, f.ufl_free_indices, f.ufl_index_dimensions)
    return dot(grad(f), FacetNormal(f.ufl_domain()))


def diff(f, v):
    """UFL operator: Take the derivative of *f* with respect to the variable *v*.

    If *f* is a form, ``diff`` is applied to each integrand.
    """
    # Apply to integrands
    if isinstance(f, Form):
        from ufl.algorithms.map_integrands import map_integrands
        return map_integrands(lambda e: diff(e, v), f)

    # Apply to expression
    f = as_ufl(f)
    if isinstance(v, SpatialCoordinate):
        return grad(f)
    elif isinstance(v, (Variable, Coefficient, ExternalOperator)):
        return VariableDerivative(f, v)
    else:
        error("Expecting a Variable or SpatialCoordinate in diff.")


def grad(f):
    """UFL operator: Take the gradient of *f*.

    This operator follows the grad convention where

      grad(s)[i] = s.dx(i)

      grad(v)[i,j] = v[i].dx(j)

      grad(T)[:,i] = T[:].dx(i)

    for scalar expressions s, vector expressions v,
    and arbitrary rank tensor expressions T.

    See also: :py:func:`nabla_grad`
    """
    f = as_ufl(f)
    return Grad(f)


def div(f):
    """UFL operator: Take the divergence of *f*.

    This operator follows the div convention where

      div(v) = v[i].dx(i)

      div(T)[:] = T[:,i].dx(i)

    for vector expressions v, and
    arbitrary rank tensor expressions T.

    See also: :py:func:`nabla_div`
    """
    f = as_ufl(f)
    return Div(f)


def nabla_grad(f):
    """UFL operator: Take the gradient of *f*.

    This operator follows the grad convention where

      nabla_grad(s)[i] = s.dx(i)

      nabla_grad(v)[i,j] = v[j].dx(i)

      nabla_grad(T)[i,:] = T[:].dx(i)

    for scalar expressions s, vector expressions v,
    and arbitrary rank tensor expressions T.

    See also: :py:func:`grad`
    """
    f = as_ufl(f)
    return NablaGrad(f)


def nabla_div(f):
    """UFL operator: Take the divergence of *f*.

    This operator follows the div convention where

      nabla_div(v) = v[i].dx(i)

      nabla_div(T)[:] = T[i,:].dx(i)

    for vector expressions v, and
    arbitrary rank tensor expressions T.

    See also: :py:func:`div`
    """
    f = as_ufl(f)
    return NablaDiv(f)


def curl(f):
    "UFL operator: Take the curl of *f*."
    f = as_ufl(f)
    return Curl(f)


rot = curl


# --- DG operators ---

def jump(v, n=None):
    "UFL operator: Take the jump of *v* across a facet."
    v = as_ufl(v)
    is_constant = len(extract_domains(v)) > 0
    if is_constant:
        if n is None:
            return v('+') - v('-')
        r = len(v.ufl_shape)
        if r == 0:
            return v('+') * n('+') + v('-') * n('-')
        else:
            return dot(v('+'), n('+')) + dot(v('-'), n('-'))
    else:
        warning("Returning zero from jump of expression without a domain. This may be erroneous if a dolfin.Expression is involved.")
        # FIXME: Is this right? If v has no domain, it doesn't depend
        # on anything spatially variable or any form arguments, and
        # thus the jump is zero. In other words, I'm assuming that "v
        # has no geometric domains" is equivalent with "v is a spatial
        # constant".  Update: This is NOT true for
        # jump(Expression("x[0]")) from dolfin.
        return Zero(v.ufl_shape, v.ufl_free_indices, v.ufl_index_dimensions)


def avg(v):
    "UFL operator: Take the average of *v* across a facet."
    v = as_ufl(v)
    return 0.5 * (v('+') + v('-'))


def cell_avg(f):
    "UFL operator: Take the average of *v* over a cell."
    return CellAvg(f)


def facet_avg(f):
    "UFL operator: Take the average of *v* over a facet."
    return FacetAvg(f)


# --- Other operators ---

def variable(e):
    """UFL operator: Define a variable representing the given expression, see also
    ``diff()``."""
    e = as_ufl(e)
    return Variable(e)


# --- Conditional expressions ---

def conditional(condition, true_value, false_value):
    """UFL operator: A conditional expression, taking the value of *true_value*
    when *condition* evaluates to ``true`` and *false_value* otherwise."""
    return Conditional(condition, true_value, false_value)


def eq(left, right):
    """UFL operator: A boolean expression (left == right) for use with
    ``conditional``."""
    return EQ(left, right)


def ne(left, right):
    """UFL operator: A boolean expression (left != right) for use with
    ``conditional``."""
    return NE(left, right)


def le(left, right):
    """UFL operator: A boolean expression (left <= right) for use with
    ``conditional``."""
    return as_ufl(left) <= as_ufl(right)


def ge(left, right):
    """UFL operator: A boolean expression (left >= right) for use with
    ``conditional``."""
    return as_ufl(left) >= as_ufl(right)


def lt(left, right):
    """UFL operator: A boolean expression (left < right) for use with
    ``conditional``."""
    return as_ufl(left) < as_ufl(right)


def gt(left, right):
    """UFL operator: A boolean expression (left > right) for use with
    ``conditional``."""
    return as_ufl(left) > as_ufl(right)


def And(left, right):
    """UFL operator: A boolean expression (left and right) for use with
    ``conditional``."""
    return AndCondition(left, right)


def Or(left, right):
    """UFL operator: A boolean expression (left or right) for use with
    ``conditional``."""
    return OrCondition(left, right)


def Not(condition):
    """UFL operator: A boolean expression (not condition) for use with
    ``conditional``."""
    return NotCondition(condition)


def sign(x):
    "UFL operator: Take the sign (+1 or -1) of *x*."
    # TODO: Add a Sign type for this?
    return conditional(eq(x, 0), 0, conditional(lt(x, 0), -1, +1))


def max_value(x, y):
    "UFL operator: Take the maximum of *x* and *y*."
    x = as_ufl(x)
    y = as_ufl(y)
    return MaxValue(x, y)


def min_value(x, y):
    "UFL operator: Take the minimum of *x* and *y*."
    x = as_ufl(x)
    y = as_ufl(y)
    return MinValue(x, y)


def Max(x, y):  # TODO: Deprecate this notation?
    "UFL operator: Take the maximum of *x* and *y*."
    return max_value(x, y)


def Min(x, y):  # TODO: Deprecate this notation?
    "UFL operator: Take the minimum of *x* and *y*."
    return min_value(x, y)


# --- Math functions ---

def _mathfunction(f, cls):
    f = as_ufl(f)
    r = cls(f)
    if isinstance(r, (RealValue, Zero, int, float)):
        return float(r)
    if isinstance(r, (ComplexValue, complex)):
        return complex(r)
    return r


def sqrt(f):
    "UFL operator: Take the square root of *f*."
    return _mathfunction(f, Sqrt)


def exp(f):
    "UFL operator: Take the exponential of *f*."
    return _mathfunction(f, Exp)


def ln(f):
    "UFL operator: Take the natural logarithm of *f*."
    return _mathfunction(f, Ln)


def cos(f):
    "UFL operator: Take the cosine of *f*."
    return _mathfunction(f, Cos)


def sin(f):
    "UFL operator: Take the sine of *f*."
    return _mathfunction(f, Sin)


def tan(f):
    "UFL operator: Take the tangent of *f*."
    return _mathfunction(f, Tan)


def cosh(f):
    "UFL operator: Take the hyperbolic cosine of *f*."
    return _mathfunction(f, Cosh)


def sinh(f):
    "UFL operator: Take the hyperbolic sine of *f*."
    return _mathfunction(f, Sinh)


def tanh(f):
    "UFL operator: Take the hyperbolic tangent of *f*."
    return _mathfunction(f, Tanh)


def acos(f):
    "UFL operator: Take the inverse cosine of *f*."
    return _mathfunction(f, Acos)


def asin(f):
    "UFL operator: Take the inverse sine of *f*."
    return _mathfunction(f, Asin)


def atan(f):
    "UFL operator: Take the inverse tangent of *f*."
    return _mathfunction(f, Atan)


def atan_2(f1, f2):
    "UFL operator: Take the inverse tangent with two the arguments *f1* and *f2*."
    f1 = as_ufl(f1)
    f2 = as_ufl(f2)
    if isinstance(f1, (ComplexValue, complex)) or isinstance(f2, (ComplexValue, complex)):
        raise TypeError('atan_2 is incompatible with complex numbers.')
    r = Atan2(f1, f2)
    if isinstance(r, (RealValue, Zero, int, float)):
        return float(r)
    if isinstance(r, (ComplexValue, complex)):
        return complex(r)
    return r


def erf(f):
    "UFL operator: Take the error function of *f*."
    return _mathfunction(f, Erf)


def bessel_J(nu, f):
    """UFL operator: cylindrical Bessel function of the first kind."""
    nu = as_ufl(nu)
    f = as_ufl(f)
    return BesselJ(nu, f)


def bessel_Y(nu, f):
    """UFL operator: cylindrical Bessel function of the second kind."""
    nu = as_ufl(nu)
    f = as_ufl(f)
    return BesselY(nu, f)


def bessel_I(nu, f):
    """UFL operator: regular modified cylindrical Bessel function."""
    nu = as_ufl(nu)
    f = as_ufl(f)
    return BesselI(nu, f)


def bessel_K(nu, f):
    """UFL operator: irregular modified cylindrical Bessel function."""
    nu = as_ufl(nu)
    f = as_ufl(f)
    return BesselK(nu, f)


# --- Special function for exterior_derivative

def exterior_derivative(f):
    """UFL operator: Take the exterior derivative of *f*.

    The exterior derivative uses the element family to
    determine whether ``id``, ``grad``, ``curl`` or ``div`` should be used.

    Note that this uses the ``grad`` and ``div`` operators,
    as opposed to ``nabla_grad`` and ``nabla_div``.
    """

    # Extract the element from the input f
    if isinstance(f, Indexed):
        expression, indices = f.ufl_operands
        if len(indices) > 1:
            raise NotImplementedError
        index = int(indices[0])
        element = expression.ufl_element()
        element = element.extract_component(index)[1]
    elif isinstance(f, ListTensor):
        f0 = f.ufl_operands[0]
        f0expr, f0indices = f0.ufl_operands  # FIXME: Assumption on type of f0!!!
        if len(f0indices) > 1:
            raise NotImplementedError
        index = int(f0indices[0])
        element = f0expr.ufl_element()
        element = element.extract_component(index)[1]
    else:
        try:
            element = f.ufl_element()
        except Exception:
            error("Unable to determine element from %s" % f)

    # Extract the family and the geometric dimension
    family = element.family()
    gdim = element.cell().geometric_dimension()

    # L^2 elements:
    if "Disc" in family:
        return f

    # H^1 elements:
    if "Lagrange" in family:
        if gdim == 1:
            return grad(f)[0]  # Special-case 1D vectors as scalars
        return grad(f)

    # H(curl) elements:
    if "curl" in family:
        return curl(f)

    # H(div) elements:
    if "Brezzi" in family or "Raviart" in family:
        return div(f)

    error("Unable to determine exterior_derivative. Family is '%s'" % family)
