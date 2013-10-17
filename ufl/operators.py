"""This module extends the form language with free function operators,
which are either already available as member functions on UFL objects
or defined as compound operators involving basic operations on the UFL
objects."""

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Kristian B. Oelgaard, 2011
#
# First added:  2008-04-09
# Last changed: 2013-03-15

import operator
from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.constantvalue import Zero, ScalarValue, as_ufl
from ufl.differentiation import VariableDerivative, Grad, Div, Curl, NablaGrad, NablaDiv
from ufl.tensoralgebra import Transposed, Inner, Outer, Dot, Cross, \
    Determinant, Inverse, Cofactor, Trace, Deviatoric, Skew, Sym
from ufl.variable import Variable
from ufl.tensors import as_tensor, as_matrix, as_vector, ListTensor
from ufl.conditional import EQ, NE, LE, GE, LT, GT, \
    AndCondition, OrCondition, NotCondition, Conditional
from ufl.mathfunctions import Sqrt, Exp, Ln, Erf,\
    Cos, Sin, Tan, Cosh, Sinh, Tanh, Acos, Asin, Atan, Atan2,\
    BesselJ, BesselY, BesselI, BesselK
from ufl.restriction import CellAvg, FacetAvg
from ufl.indexing import indices
from ufl.indexed import Indexed
from ufl.geometry import SpatialCoordinate

#--- Basic operators ---

def rank(f):
    "UFL operator: The rank of f."
    f = as_ufl(f)
    return len(f.shape())

def shape(f):
    "UFL operator: The shape of f."
    f = as_ufl(f)
    return f.shape()

#--- Elementwise tensor operators ---

def elem_op_items(op_ind, indices, *args):
    sh = args[0].shape()
    n = sh[len(indices)]
    def extind(ii):
        return tuple(list(indices) + [ii])
    if len(sh) == len(indices)+1:
        return [op_ind(extind(i), *args) for i in range(n)]
    else:
        return [elem_op_items(op_ind, extind(i), *args) for i in range(n)]

def elem_op(op, *args):
    "UFL operator: Take the elementwise application of operator op on scalar values from one or more tensor arguments."
    args = map(as_ufl, args)
    sh = args[0].shape()
    ufl_assert(all(sh == x.shape() for x in args),
               "Cannot take elementwise operation with different shapes.")
    if sh == ():
        return op(*args)
    def op_ind(ind, *args):
        return op(*[x[ind] for x in args])
    return as_tensor(elem_op_items(op_ind, (), *args))

def elem_mult(A, B):
    "UFL operator: Take the elementwise multiplication of the tensors A and B with the same shape."
    return elem_op(operator.mul, A, B)

def elem_div(A, B):
    "UFL operator: Take the elementwise division of the tensors A and B with the same shape."
    return elem_op(operator.div, A, B)

def elem_pow(A, B):
    "UFL operator: Take the elementwise power of the tensors A and B with the same shape."
    return elem_op(operator.pow, A, B)


#--- Tensor operators ---

def transpose(A):
    "UFL operator: Take the transposed of tensor A."
    A = as_ufl(A)
    if A.shape() == ():
        return A
    return Transposed(A)

def outer(*operands):
    "UFL operator: Take the outer product of two or more operands."
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
    if a.shape() == () and b.shape() == ():
        return a*b
    return Outer(a, b)

def inner(a, b):
    "UFL operator: Take the inner product of a and b."
    a = as_ufl(a)
    b = as_ufl(b)
    if a.shape() == () and b.shape() == ():
        return a*b
    return Inner(a, b)

# TODO: Something like this would be useful in some cases,
# but should inner just support a.rank() != b.rank() instead?
def _partial_inner(a, b):
    "UFL operator: Take the partial inner product of a and b."
    ar, br = a.rank(), b.rank()
    n = min(ar, br)
    return contraction(a, range(n-ar, n-ar+n), b, range(n))

def dot(a, b):
    "UFL operator: Take the dot product of a and b."
    a = as_ufl(a)
    b = as_ufl(b)
    if a.shape() == () and b.shape() == ():
        return a*b
    return Dot(a, b)
    #return contraction(a, (a.rank()-1,), b, (b.rank()-1,))

def contraction(a, a_axes, b, b_axes):
    "UFL operator: Take the contraction of a and b over given axes."
    ai, bi = a_axes, b_axes
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

def perp(v):
    "UFL operator: Take the perp of v, i.e. (-v1, +v0)."
    v = as_ufl(v)
    ufl_assert(v.shape() == (2,), "Expecting a 2D vector expression.")
    return as_vector((-v[1],v[0]))

def cross(a, b):
    "UFL operator: Take the cross product of a and b."
    a = as_ufl(a)
    b = as_ufl(b)
    #ufl_assert(a.shape() == (3,) and b.shape() == (3,),
    #           "Expecting 3D vectors in cross product.")
    return Cross(a, b)

def det(A):
    "UFL operator: Take the determinant of A."
    A = as_ufl(A)
    if A.shape() == ():
        return A
    return Determinant(A)

def inv(A):
    "UFL operator: Take the inverse of A."
    A = as_ufl(A)
    if A.shape() == ():
        return 1 / A
    return Inverse(A)

def cofac(A):
    "UFL operator: Take the cofactor of A."
    A = as_ufl(A)
    return Cofactor(A)

def tr(A):
    "UFL operator: Take the trace of A."
    A = as_ufl(A)
    return Trace(A)

def diag(A):
    """UFL operator: Take the diagonal part of rank 2 tensor A _or_
    make a diagonal rank 2 tensor from a rank 1 tensor.

    Always returns a rank 2 tensor. See also diag_vector."""

    # TODO: Make a compound type or two for this operator

    # Get and check dimensions
    r = A.rank()
    if r == 1:
        n, = A.shape()
    elif r == 2:
        m, n = A.shape()
        ufl_assert(m == n, "Can only take diagonal of square tensors.")
    else:
        error("Expecting rank 1 or 2 tensor.")

    # Build matrix row by row
    rows = []
    for i in range(n):
        row = [0]*n
        row[i] = A[i] if r == 1 else A[i,i]
        rows.append(row)
    return as_matrix(rows)

def diag_vector(A):
    """UFL operator: Take the diagonal part of rank 2 tensor A and return as a vector.

    See also diag."""

    # TODO: Make a compound type for this operator

    # Get and check dimensions
    ufl_assert(A.rank() == 2, "Expecting rank 2 tensor.")
    m, n = A.shape()
    ufl_assert(m == n, "Can only take diagonal of square tensors.")

    # Return diagonal vector
    return as_vector([A[i,i] for i in range(n)])

def dev(A):
    "UFL operator: Take the deviatoric part of A."
    A = as_ufl(A)
    return Deviatoric(A)

def skew(A):
    "UFL operator: Take the skew symmetric part of A."
    A = as_ufl(A)
    return Skew(A)

def sym(A):
    "UFL operator: Take the symmetric part of A."
    A = as_ufl(A)
    return Sym(A)

#--- Differential operators

def Dx(f, *i):
    """UFL operator: Take the partial derivative of f with respect
    to spatial variable number i. Equivalent to f.dx(\*i)."""
    f = as_ufl(f)
    return f.dx(*i)

def Dt(f):
    "UFL operator: <Not implemented yet!> The partial derivative of f with respect to time."
    #return TimeDerivative(f) # TODO: Add class
    raise NotImplementedError

def Dn(f):
    """UFL operator: Take the directional derivative of f in the
    facet normal direction, Dn(f) := dot(grad(f), n)."""
    f = as_ufl(f)
    cell = f.cell()
    if cell is None: # FIXME: Rather if f.is_cellwise_constant()?
        return Zero(f.shape(), f.free_indices(), f.index_dimensions())
    return dot(grad(f), cell.n)

def diff(f, v):
    """UFL operator: Take the derivative of f with respect to the variable v.

    If f is a form, diff is applied to each integrand.
    """
    if isinstance(f, Form):
        from ufl.algorithms.transformer import transform_integrands
        return transform_integrands(f, lambda e: diff(e, v))
    else:
        f = as_ufl(f)

    if isinstance(v, SpatialCoordinate):
        return grad(f)
    # TODO: Allow this? Must be tested well!
    #elif (isinstance(v, Indexed)
    #      and isinstance(v.operands()[0], SpatialCoordinate)):
    #    return grad(f)[...,v.operands()[1]]
    elif isinstance(v, Variable):
        return VariableDerivative(f, v)
    else:
        error("Expecting a Variable or SpatialCoordinate in diff.")

def grad(f):
    """UFL operator: Take the gradient of f.

    This operator follows the grad convention where

      grad(s)[i] = s.dx(j)

      grad(v)[i,j] = v[i].dx(j)

      grad(T)[:,i] = T[:].dx(i)

    for scalar expressions s, vector expressions v,
    and arbitrary rank tensor expressions T.
    """
    f = as_ufl(f)
    return Grad(f)

def div(f):
    """UFL operator: Take the divergence of f.

    This operator follows the div convention where

      div(v) = v[i].dx(i)

      div(T)[:] = T[:,i].dx(i)

    for vector expressions v, and
    arbitrary rank tensor expressions T.
    """
    f = as_ufl(f)
    return Div(f)

def nabla_grad(f):
    """UFL operator: Take the gradient of f.

    This operator follows the grad convention where

      nabla_grad(s)[i] = s.dx(j)

      nabla_grad(v)[i,j] = v[j].dx(i)

      nabla_grad(T)[i,:] = T[:].dx(i)

    for scalar expressions s, vector expressions v,
    and arbitrary rank tensor expressions T.
    """
    f = as_ufl(f)
    return NablaGrad(f)

def nabla_div(f):
    """UFL operator: Take the divergence of f.

    This operator follows the div convention where

      nabla_div(v) = v[i].dx(i)

      nabla_div(T)[:] = T[i,:].dx(i)

    for vector expressions v, and
    arbitrary rank tensor expressions T.
    """
    f = as_ufl(f)
    return NablaDiv(f)

def curl(f):
    "UFL operator: Take the curl of f."
    f = as_ufl(f)
    return Curl(f)
rot = curl

#--- DG operators ---

def jump(v, n=None):
    "UFL operator: Take the jump of v across a facet."
    v = as_ufl(v)
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
        else:
            return dot(v('+'), n('+')) + dot(v('-'), n('-'))

def avg(v):
    "UFL operator: Take the average of v across a facet."
    v = as_ufl(v)
    return 0.5*(v('+') + v('-'))

def cell_avg(f):
    "UFL operator: Take the average of v over a cell."
    #ufl_assert((isinstance(f, Restricted) and isinstance(f.operands()[0], FormArgument)) or
    #    isinstance(f, FormArgument), "Can only take the cell average of a (optionally restricted) Coefficient or Argument.")
    return CellAvg(f)

def facet_avg(f):
    "UFL operator: Take the average of v over a facet."
    #ufl_assert((isinstance(f, Restricted) and isinstance(f.operands()[0], FormArgument)) or
    #    isinstance(f, FormArgument), "Can only take the cell average of a (optionally restricted) Coefficient or Argument.")
    return FacetAvg(f)

#--- Other operators ---

def variable(e):
    "UFL operator: Define a variable representing the given expression, see also diff()."
    e = as_ufl(e)
    return Variable(e)

#--- Conditional expressions ---

def conditional(condition, true_value, false_value):
    """UFL operator: A conditional expression, taking the value of true_value
    when condition evaluates to true and false_value otherwise."""
    return Conditional(condition, true_value, false_value)

def eq(left, right):
    "UFL operator: A boolean expresion (left == right) for use with conditional."
    return EQ(left, right)

def ne(left, right):
    "UFL operator: A boolean expresion (left != right) for use with conditional."
    return NE(left, right)

def le(left, right):
    "UFL operator: A boolean expresion (left <= right) for use with conditional."
    return as_ufl(left) <= as_ufl(right)

def ge(left, right):
    "UFL operator: A boolean expresion (left >= right) for use with conditional."
    return as_ufl(left) >= as_ufl(right)

def lt(left, right):
    "UFL operator: A boolean expresion (left < right) for use with conditional."
    return as_ufl(left) < as_ufl(right)

def gt(left, right):
    "UFL operator: A boolean expresion (left > right) for use with conditional."
    return as_ufl(left) > as_ufl(right)

def And(left, right):
    "UFL operator: A boolean expresion (left and right) for use with conditional."
    return AndCondition(left, right)

def Or(left, right):
    "UFL operator: A boolean expresion (left or right) for use with conditional."
    return OrCondition(left, right)

def Not(condition):
    "UFL operator: A boolean expresion (not condition) for use with conditional."
    return NotCondition(condition)

def sign(x):
    "UFL operator: Take the sign (+1 or -1) of x."
    # TODO: Add a Sign type for this?
    return conditional(eq(x, 0), 0, conditional(lt(x, 0), -1, +1))

def Max(x, y):
    "UFL operator: Take the maximum of x and y."
    # TODO: Add a Maximum type for this?
    return conditional(gt(x, y), x, y)

def Min(x, y):
    "UFL operator: Take the minimum of x and y."
    # TODO: Add a Minimum type for this?
    return conditional(lt(x, y), x, y)

#--- Math functions ---

def _mathfunction(f, cls):
    f = as_ufl(f)
    r = cls(f)
    if isinstance(r, (ScalarValue, Zero, int, float)):
        return float(r)
    return r

def sqrt(f):
    "UFL operator: Take the square root of f."
    return _mathfunction(f, Sqrt)

def exp(f):
    "UFL operator: Take the exponential of f."
    return _mathfunction(f, Exp)

def ln(f):
    "UFL operator: Take the natural logarithm of f."
    return _mathfunction(f, Ln)

def cos(f):
    "UFL operator: Take the cosinus of f."
    return _mathfunction(f, Cos)

def sin(f):
    "UFL operator: Take the sinus of f."
    return _mathfunction(f, Sin)

def tan(f):
    "UFL operator: Take the tangent of f."
    return _mathfunction(f, Tan)

def cosh(f):
    "UFL operator: Take the cosinus hyperbolicus of f."
    return _mathfunction(f, Cosh)

def sinh(f):
    "UFL operator: Take the sinus hyperbolicus of f."
    return _mathfunction(f, Sinh)

def tanh(f):
    "UFL operator: Take the tangent hyperbolicus of f."
    return _mathfunction(f, Tanh)

def acos(f):
    "UFL operator: Take the inverse cosinus of f."
    return _mathfunction(f, Acos)

def asin(f):
    "UFL operator: Take the inverse sinus of f."
    return _mathfunction(f, Asin)

def atan(f):
    "UFL operator: Take the inverse tangent of f."
    return _mathfunction(f, Atan)

def atan_2(f1,f2):
    "UFL operator: Take the inverse tangent of f."
    f1 = as_ufl(f1)
    f2 = as_ufl(f2)
    r = Atan2(f1, f2)
    if isinstance(r, (ScalarValue, Zero, int, float)):
        return float(r)
    return r

def erf(f):
    "UFL operator: Take the error function of f."
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


#--- Special function for exterior_derivative

def exterior_derivative(f):
    """UFL operator: Take the exterior derivative of f.

    The exterior derivative uses the element family to
    determine whether id, grad, curl or div should be used.

    Note that this uses the 'grad' and 'div' operators,
    as opposed to 'nabla_grad' and 'nabla_div'.
    """

    # Extract the element from the input f
    if isinstance(f, Indexed):
        if len(f._indices) > 1:
            raise NotImplementedError
        index = int(f._indices[0])
        element = f._expression.element()
        element = element.extract_component(index)[1]
    elif isinstance(f, ListTensor):
        if len(f._expressions[0]._indices) > 1:
            raise NotImplementedError
        index = int(f._expressions[0]._indices[0])
        element = f._expressions[0]._expression.element()
        element = element.extract_component(index)[1]
    else:
        try:
            element = f.element()
        except:
            error("Unable to determine element from %s" % f)

    # Extract the family and the
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
