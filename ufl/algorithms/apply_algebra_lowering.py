"""Algorithm for expanding compound expressions into
equivalent representations using basic operators."""

# Copyright (C) 2008-2015 Martin Sandve Alnes and Anders Logg
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

from ufl.classes import Product, Grad
from ufl.core.multiindex import indices, Index, FixedIndex
from ufl.tensors import as_tensor, as_matrix, as_vector

from ufl.compound_expressions import deviatoric_expr, determinant_expr, cofactor_expr, adj_expr, inverse_expr

from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags


class LowerCompoundAlgebra(MultiFunction):
    """Expands high level compound operators (e.g. inner) to equivalent
    representations using basic operators (e.g. index notation)."""
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

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

    def altenative_dot(self, o, a, b): # TODO: Test this
        ash = a.ufl_shape
        bsh = b.ufl_shape
        ai = indices(len(ash)-1)
        bi = indices(len(bsh)-1)
        # Simplification for tensors where the dot-sum dimension has length 1
        if ash[-1] == 1:
            k = (FixedIndex(0),)
        else:
            k = (Index(),)
        # Potentially creates a single IndexSum over a Product
        s = a[ai+k]*b[k+bi]
        return as_tensor(s, ai+bi)

    def dot(self, o, a, b):
        ai = indices(a.rank()-1)
        bi = indices(b.rank()-1)
        k = (Index(),)
        # Creates a single IndexSum over a Product
        s = a[ai+k]*b[k+bi]
        return as_tensor(s, ai+bi)

    def alternative_inner(self, o, a, b): # TODO: Test this
        ash = a.ufl_shape
        bsh = b.ufl_shape
        ufl_assert(ash == bsh)
        # Simplification for tensors with one or more dimensions of length 1
        ii = []
        zi = FixedIndex(0)
        for n in ash:
            if n == 1:
                ii.append(zi)
            else:
                ii.append(Index())
        ii = tuple(ii)
        #ii = indices(a.rank())
        # Potentially creates multiple IndexSums over a Product
        s = a[ii]*b[ii]
        return s

    def inner(self, o, a, b):
        ufl_assert(a.ufl_shape == b.ufl_shape)
        ii = indices(a.rank())
        # Creates multiple IndexSums over a Product
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


def apply_algebra_lowering(expr):
    """Expands high level compound operators (e.g. inner) to equivalent
    representations using basic operators (e.g. index notation)."""
    return map_integrand_dags(LowerCompoundAlgebra(), expr)
