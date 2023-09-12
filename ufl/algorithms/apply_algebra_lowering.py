# -*- coding: utf-8 -*-
"""Algorithm for expanding compound expressions into
equivalent representations using basic operators."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

from ufl.classes import Product, Grad, Conj
from ufl.core.multiindex import indices, Index
from ufl.tensors import as_tensor, as_matrix, as_vector

from ufl.compound_expressions import deviatoric_expr, determinant_expr, cofactor_expr, inverse_expr

from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags


class LowerCompoundAlgebra(MultiFunction):
    """Expands high level compound operators (e.g. inner) to equivalent
    representations using basic operators (e.g. index notation)."""

    def __init__(self):
        MultiFunction.__init__(self)

    ufl_type = MultiFunction.reuse_if_untouched

    # ------------ Compound tensor operators

    def trace(self, o, A):
        i = Index()
        return A[i, i]

    def transposed(self, o, A):
        i, j = indices(2)
        return as_tensor(A[i, j], (j, i))

    def deviatoric(self, o, A):
        return deviatoric_expr(A)

    def skew(self, o, A):
        i, j = indices(2)
        return as_matrix((A[i, j] - A[j, i]) / 2, (i, j))

    def sym(self, o, A):
        i, j = indices(2)
        return as_matrix((A[i, j] + A[j, i]) / 2, (i, j))

    def cross(self, o, a, b):
        def c(i, j):
            return Product(a[i], b[j]) - Product(a[j], b[i])
        return as_vector((c(1, 2), c(2, 0), c(0, 1)))

    def perp(self, o, a):
        return as_vector([-a[1], a[0]])

    def dot(self, o, a, b):
        ai = indices(len(a.ufl_shape) - 1)
        bi = indices(len(b.ufl_shape) - 1)
        k = (Index(),)
        # Creates a single IndexSum over a Product
        s = a[ai + k] * b[k + bi]
        return as_tensor(s, ai + bi)

    def inner(self, o, a, b):
        ash = a.ufl_shape
        bsh = b.ufl_shape
        if ash != bsh:
            raise ValueError("Nonmatching shapes.")
        ii = indices(len(ash))
        # Creates multiple IndexSums over a Product
        s = a[ii] * Conj(b[ii])
        return s

    def outer(self, o, a, b):
        ii = indices(len(a.ufl_shape))
        jj = indices(len(b.ufl_shape))
        # Create a Product with no shared indices
        s = Conj(a[ii]) * b[jj]
        return as_tensor(s, ii + jj)

    def determinant(self, o, A):
        return determinant_expr(A)

    def cofactor(self, o, A):
        return cofactor_expr(A)

    def inverse(self, o, A):
        return inverse_expr(A)

    # ------------ Compound differential operators

    def div(self, o, a):
        i = Index()
        return a[..., i].dx(i)

    def nabla_div(self, o, a):
        i = Index()
        return a[i, ...].dx(i)

    def nabla_grad(self, o, a):
        sh = a.ufl_shape
        if sh == ():
            return Grad(a)
        else:
            j = Index()
            ii = tuple(indices(len(sh)))
            return as_tensor(a[ii].dx(j), (j,) + ii)

    def curl(self, o, a):
        # o = curl a = "[a.dx(1), -a.dx(0)]"            if a.ufl_shape == ()
        # o = curl a = "cross(nabla, (a0, a1, 0))[2]" if a.ufl_shape == (2,)
        # o = curl a = "cross(nabla, a)"              if a.ufl_shape == (3,)
        def c(i, j):
            return a[j].dx(i) - a[i].dx(j)
        sh = a.ufl_shape
        if sh == ():
            return as_vector((a.dx(1), -a.dx(0)))
        if sh == (2,):
            return c(0, 1)
        if sh == (3,):
            return as_vector((c(1, 2), c(2, 0), c(0, 1)))
        raise ValueError(f"Invalid shape {sh} of curl argument.")


def apply_algebra_lowering(expr):
    """Expands high level compound operators (e.g. inner) to equivalent
    representations using basic operators (e.g. index notation)."""
    return map_integrand_dags(LowerCompoundAlgebra(), expr)
