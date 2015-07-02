"""Algorithm for expanding compound expressions into
equivalent representations using basic operators."""

# Copyright (C) 2008-2014 Martin Sandve Alnes and Anders Logg
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
# Modified by Anders Logg, 2009-2010

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.classes import Product, Index, Zero, FormArgument, Grad
from ufl.core.multiindex import indices
from ufl.tensors import as_tensor, as_matrix, as_vector
from ufl.algorithms.transformer import Transformer, ReuseTransformer, apply_transformer
from ufl.compound_expressions import deviatoric_expr, determinant_expr, cofactor_expr, adj_expr, inverse_expr

class CompoundExpander(ReuseTransformer):
    "Expands compound expressions to equivalent representations using basic operators."
    def __init__(self):
        ReuseTransformer.__init__(self)

    # ------------ Compound tensor operators

    def trace(self, o, A):
        i = Index()
        return A[i, i]

    def transposed(self, o, A):
        i, j = indices(2)
        return as_tensor(A[i, j], (j, i))

    def _square_matrix_shape(self, A):
        sh = A.ufl_shape
        ufl_assert(sh[0] == sh[1], "Expecting square matrix.")
        ufl_assert(sh[0] is not None, "Unknown dimension.")
        return sh

    def deviatoric(self, o, A):
        return deviatoric_expr(A)

    def skew(self, o, A):
        i, j = indices(2)
        return as_matrix( (A[i, j] - A[j, i]) / 2, (i, j) )

    def sym(self, o, A):
        i, j = indices(2)
        return as_matrix( (A[i, j] + A[j, i]) / 2, (i, j) )

    def cross(self, o, a, b):
        def c(i, j):
            return Product(a[i], b[j]) - Product(a[j], b[i])
        return as_vector((c(1, 2), c(2, 0), c(0, 1)))

    def dot(self, o, a, b):
        ai = indices(a.rank()-1)
        bi = indices(b.rank()-1)
        k  = indices(1)
        # Create an IndexSum over a Product
        s = a[ai+k]*b[k+bi]
        return as_tensor(s, ai+bi)

    def inner(self, o, a, b):
        ufl_assert(a.rank() == b.rank())
        ii = indices(a.rank())
        # Create multiple IndexSums over a Product
        s = a[ii]*b[ii]
        return s

    def outer(self, o, a, b):
        ii = indices(a.rank())
        jj = indices(b.rank())
        # Create a Product with no shared indices
        s = a[ii]*b[jj]
        return as_tensor(s, ii+jj)

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

    def grad(self, o, a):
        return self.reuse_if_possible(o, a)

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
        error("Invalid shape %s of curl argument." % (sh,))


"""
FIXME: Make expand_compounds_prediff skip types that we make
work in expand_derivatives, one by one, and optionally
use it instead of expand_compounds from expand_derivatives.
"""

class CompoundExpanderPreDiff(CompoundExpander):
    def __init__(self):
        CompoundExpander.__init__(self)

    #inner = Transformer.reuse_if_possible
    #dot = Transformer.reuse_if_possible

    def grad(self, o, a):
        return self.reuse_if_possible(o, a)

    def nabla_grad(self, o, a):
        r = o.rank()
        ii = indices(r)
        jj = ii[-1:] + ii[:-1]
        return as_tensor(Grad(a)[ii], jj)

    def div(self, o, a):
        i = Index()
        return Grad(a)[..., i, i]

    def nabla_div(self, o, a):
        i = Index()
        return Grad(a)[i, ..., i]

    def curl(self, o, a):
        # o = curl a = "[a.dx(1), -a.dx(0)]"            if a.ufl_shape == ()
        # o = curl a = "cross(nabla, (a0, a1, 0))[2]" if a.ufl_shape == (2,)
        # o = curl a = "cross(nabla, a)"              if a.ufl_shape == (3,)
        Da = Grad(a)
        def c(i, j):
            #return a[j].dx(i) - a[i].dx(j)
            return Da[j, i] - Da[i, j]
        sh = a.ufl_shape
        if sh == ():
            #return as_vector((a.dx(1), -a.dx(0)))
            return as_vector((Da[1], -Da[0]))
        if sh == (2,):
            return c(0, 1)
        if sh == (3,):
            return as_vector((c(1, 2), c(2, 0), c(0, 1)))
        error("Invalid shape %s of curl argument." % (sh,))

class CompoundExpanderPostDiff(CompoundExpander):
    def __init__(self):
        CompoundExpander.__init__(self)

    def nabla_grad(self, o, a, i):
        error("This should not happen.")

    def div(self, o, a, i):
        error("This should not happen.")

    def nabla_div(self, o, a, i):
        error("This should not happen.")

    def curl(self, o, a, i):
        error("This should not happen.")

def expand_compounds1(e):
    """Expand compound objects into basic operators.
    Requires e to have a well defined geometric dimension."""
    return apply_transformer(e, CompoundExpander())

def expand_compounds2(e):
    """Expand compound objects into basic operators.
    Requires e to have a well defined geometric dimension."""
    return expand_compounds_postdiff(expand_compounds_prediff(e))

def expand_compounds_prediff(e):
    """Expand compound objects into basic operators.
    Requires e to have a well defined geometric dimension."""
    return apply_transformer(e, CompoundExpanderPreDiff())

def expand_compounds_postdiff(e):
    """Expand compound objects into basic operators.
    Requires e to have a well defined geometric dimension."""
    return apply_transformer(e, CompoundExpanderPostDiff())

expand_compounds = expand_compounds1
