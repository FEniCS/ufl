# -*- coding: utf-8 -*-
"""This module attaches special functions to Expr.
This way we avoid circular dependencies between e.g.
Sum and its superclass Expr."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016.

import numbers

from ufl.algebra import Abs, Division, Power, Product, Sum
from ufl.conditional import GE, GT, LE, LT
from ufl.constantvalue import Zero, as_ufl
from ufl.core.expr import Expr
from ufl.core.multiindex import Index, MultiIndex, indices
from ufl.differentiation import Grad
from ufl.exprequals import expr_equals
from ufl.index_combination_utils import (create_slice_indices,
                                         merge_overlapping_indices)
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.restriction import NegativeRestricted, PositiveRestricted
from ufl.tensoralgebra import Inner, Transposed
from ufl.tensors import ComponentTensor, as_tensor
from ufl.utils.stacks import StackDict

# --- Boolean operators ---


def _le(left, right):
    "UFL operator: A boolean expresion (left <= right) for use with conditional."
    return LE(left, right)


def _ge(left, right):
    "UFL operator: A boolean expresion (left >= right) for use with conditional."
    return GE(left, right)


def _lt(left, right):
    "UFL operator: A boolean expresion (left < right) for use with conditional."
    return LT(left, right)


def _gt(left, right):
    "UFL operator: A boolean expresion (left > right) for use with conditional."
    return GT(left, right)


# '==' needs to implement comparison of expression representations for
# use in hashmaps (dict and set), but the others can be overloaded in
# the language.  It is possible that we can overload eq as well, but
# we'll need to fix some issues first and also check for a possible
# significant performance hit with compilation of complex
# forms. Replacing a==b with equiv(a,b) all over the code could be one
# way to reduce such a performance hit, but we cannot do anything
# about dict and set calling __eq__...
Expr.__eq__ = expr_equals


# != is used at least by tests, possibly in code as well, and must
# mean the opposite of ==, i.e. when evaluated as bool it must mean
# 'not equal representation'.
def _ne(self, other):
    return not self.__eq__(other)


Expr.__ne__ = _ne
Expr.__lt__ = _lt
Expr.__gt__ = _gt
Expr.__le__ = _le
Expr.__ge__ = _ge

# Python operators 'and'/'or' cannot be overloaded, and bitwise
# operators &/| don't have the right precedence levels
# Expr.__and__ = _and
# Expr.__or__ = _or


def _as_tensor(self, indices):
    "UFL operator: A^indices := as_tensor(A, indices)."
    if not isinstance(indices, tuple):
        raise ValueError("Expecting a tuple of Index objects to A^indices := as_tensor(A, indices).")
    if not all(isinstance(i, Index) for i in indices):
        raise ValueError("Expecting a tuple of Index objects to A^indices := as_tensor(A, indices).")
    return as_tensor(self, indices)


Expr.__xor__ = _as_tensor


# --- Helper functions for product handling ---

def _mult(a, b):
    # Discover repeated indices, which results in index sums
    afi = a.ufl_free_indices
    bfi = b.ufl_free_indices
    afid = a.ufl_index_dimensions
    bfid = b.ufl_index_dimensions
    fi, fid, ri, rid = merge_overlapping_indices(afi, afid, bfi, bfid)

    # Pick out valid non-scalar products here (dot products):
    # - matrix-matrix (A*B, M*grad(u)) => A . B
    # - matrix-vector (A*v) => A . v
    s1, s2 = a.ufl_shape, b.ufl_shape
    r1, r2 = len(s1), len(s2)

    if r1 == 0 and r2 == 0:
        # Create scalar product
        p = Product(a, b)
        ti = ()

    elif r1 == 0 or r2 == 0:
        # Scalar - tensor product
        if r2 == 0:
            a, b = b, a

        # Check for zero, simplifying early if possible
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = s1 or s2
            return Zero(shape, fi, fid)

        # Repeated indices are allowed, like in:
        # v[i]*M[i,:]

        # Apply product to scalar components
        ti = indices(len(b.ufl_shape))
        p = Product(a, b[ti])

    elif r1 == 2 and r2 in (1, 2):  # Matrix-matrix or matrix-vector
        if ri:
            raise ValueError("Not expecting repeated indices in non-scalar product.")

        # Check for zero, simplifying early if possible
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = s1[:-1] + s2[1:]
            return Zero(shape, fi, fid)

        # Return dot product in index notation
        ai = indices(len(a.ufl_shape) - 1)
        bi = indices(len(b.ufl_shape) - 1)
        k = indices(1)

        p = a[ai + k] * b[k + bi]
        ti = ai + bi

    else:
        raise ValueError(f"Invalid ranks {r1} and {r2} in product.")

    # TODO: I think applying as_tensor after index sums results in
    # cleaner expression graphs.
    # Wrap as tensor again
    if ti:
        p = as_tensor(p, ti)

    # If any repeated indices were found, apply implicit summation
    # over those
    for i in ri:
        mi = MultiIndex((Index(count=i),))
        p = IndexSum(p, mi)

    return p


# --- Extend Expr with algebraic operators ---

_valid_types = (Expr, numbers.Real, numbers.Integral, numbers.Complex)


def _mul(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    o = as_ufl(o)
    return _mult(self, o)


Expr.__mul__ = _mul


def _rmul(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    o = as_ufl(o)
    return _mult(o, self)


Expr.__rmul__ = _rmul


def _add(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(self, o)


Expr.__add__ = _add


def _radd(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    if isinstance(o, numbers.Number) and o == 0:
        # Allow adding scalar int 0 as a no-op, even for shaped self,
        # needed for sum([a,b])
        return self
    return Sum(o, self)


Expr.__radd__ = _radd


def _sub(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(self, -o)


Expr.__sub__ = _sub


def _rsub(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(o, -self)


Expr.__rsub__ = _rsub


def _div(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    sh = self.ufl_shape
    if sh:
        ii = indices(len(sh))
        d = Division(self[ii], o)
        return as_tensor(d, ii)
    return Division(self, o)


Expr.__div__ = _div
Expr.__truediv__ = _div


def _rdiv(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Division(o, self)


Expr.__rdiv__ = _rdiv
Expr.__rtruediv__ = _rdiv


def _pow(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    if o == 2 and self.ufl_shape:
        return Inner(self, self)
    return Power(self, o)


Expr.__pow__ = _pow


def _rpow(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Power(o, self)


Expr.__rpow__ = _rpow


# TODO: Add Negated class for this? Might simplify reductions in Add.
def _neg(self):
    return -1 * self


Expr.__neg__ = _neg


def _abs(self):
    return Abs(self)


Expr.__abs__ = _abs


# --- Extend Expr with restiction operators a("+"), a("-") ---

def _restrict(self, side):
    if side == "+":
        return PositiveRestricted(self)
    if side == "-":
        return NegativeRestricted(self)
    raise ValueError(f"Invalid side '{side}' in restriction operator.")


def _eval(self, coord, mapping=None, component=()):
    # Evaluate expression at this particular coordinate, with provided
    # values for other terminals in mapping

    # Evaluate derivatives first
    from ufl.algorithms import expand_derivatives
    f = expand_derivatives(self)

    # Evaluate recursively
    if mapping is None:
        mapping = {}
    index_values = StackDict()
    return f.evaluate(coord, mapping, component, index_values)


def _call(self, arg, mapping=None, component=()):
    # Taking the restriction or evaluating depending on argument
    if arg in ("+", "-"):
        if mapping is not None:
            raise ValueError("Not expecting a mapping when taking restriction.")
        return _restrict(self, arg)
    else:
        return _eval(self, arg, mapping, component)


Expr.__call__ = _call


# --- Extend Expr with the transpose operation A.T ---

def _transpose(self):
    """Transpose a rank-2 tensor expression. For more general transpose
    operations of higher order tensor expressions, use indexing and Tensor."""
    return Transposed(self)


Expr.T = property(_transpose)


# --- Extend Expr with indexing operator a[i] ---

def _getitem(self, component):

    # Treat component consistently as tuple below
    if not isinstance(component, tuple):
        component = (component,)

    shape = self.ufl_shape

    # Analyse slices (:) and Ellipsis (...)
    all_indices, slice_indices, repeated_indices = create_slice_indices(component, shape, self.ufl_free_indices)

    # Check that we have the right number of indices for a tensor with
    # this shape
    if len(shape) != len(all_indices):
        raise ValueError(f"Invalid number of indices {len(all_indices)} for expression of rank {len(shape)}.")

    # Special case for simplifying foo[...] => foo, foo[:] => foo or
    # similar
    if len(slice_indices) == len(all_indices):
        return self

    # Special case for simplifying as_tensor(ai,(i,))[i] => ai
    if isinstance(self, ComponentTensor):
        if all_indices == self.indices().indices():
            return self.ufl_operands[0]

    # Apply all indices to index self, yielding a scalar valued
    # expression
    mi = MultiIndex(all_indices)
    a = Indexed(self, mi)

    # TODO: I think applying as_tensor after index sums results in
    # cleaner expression graphs.

    # If the Ellipsis or any slices were found, wrap as tensor valued
    # with the slice indices created at the top here
    if slice_indices:
        a = as_tensor(a, slice_indices)

    # If any repeated indices were found, apply implicit summation
    # over those
    for i in repeated_indices:
        mi = MultiIndex((i,))
        a = IndexSum(a, mi)

    # Check for zero (last so we can get indices etc from a, could
    # possibly be done faster by checking early instead)
    if isinstance(self, Zero):
        shape = a.ufl_shape
        fi = a.ufl_free_indices
        fid = a.ufl_index_dimensions
        a = Zero(shape, fi, fid)

    return a


Expr.__getitem__ = _getitem


# --- Extend Expr with spatial differentiation operator a.dx(i) ---

def _dx(self, *ii):
    "Return the partial derivative with respect to spatial variable number *ii*."
    d = self
    # Unwrap ii to allow .dx(i,j) and .dx((i,j))
    if len(ii) == 1 and isinstance(ii[0], tuple):
        ii = ii[0]
    # Apply all derivatives
    for i in ii:
        d = Grad(d)

    # Take all components, applying repeated index sums in the [] operation
    return d.__getitem__((Ellipsis,) + ii)


Expr.dx = _dx
