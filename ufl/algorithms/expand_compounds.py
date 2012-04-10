"""Algorithm for expanding compound expressions into
equivalent representations using basic operators."""

# Copyright (C) 2008-2012 Martin Sandve Alnes and Anders Logg
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
#
# First added:  2008-05-07
# Last changed: 2012-04-10

# TODO: Remove unused imports here

from itertools import izip, chain
from inspect import getargspec

from ufl.log import error, warning, debug, info
from ufl.common import Stack, StackDict
from ufl.assertions import ufl_assert
from ufl.classes import Expr, Terminal, Product, Index, FixedIndex, ListTensor, Variable, Zero, CoefficientDerivative
from ufl.indexing import indices, complete_shape
from ufl.tensors import as_tensor, as_matrix, as_vector
from ufl.form import Form
from ufl.integral import Integral
from ufl.classes import all_ufl_classes
from ufl.algorithms.analysis import has_type, extract_type, extract_duplications
from ufl.constantvalue import as_ufl

from ufl.algorithms.transformer import Transformer, ReuseTransformer, apply_transformer


# Note: To avoid typing errors, the expressions for cofactor and
# deviatoric parts below were created with the script
# tensoralgebrastrings.py under sandbox/scripts/

class CompoundExpander(ReuseTransformer):
    "Expands compound expressions to equivalent representations using basic operators."
    def __init__(self, geometric_dimension):
        ReuseTransformer.__init__(self)
        self._dim = geometric_dimension

        #if self._dim is None:
        #    warning("Got None for dimension, some compounds cannot be expanded.")

    # ------------ Compound tensor operators

    def trace(self, o, A):
        i = Index()
        return A[i,i]

    def transposed(self, o, A):
        i, j = indices(2)
        return as_tensor(A[i, j], (j, i))

    def _square_matrix_shape(self, A):
        sh = A.shape()
        if self._dim is not None:
            sh = complete_shape(sh, self._dim)
        ufl_assert(sh[0] == sh[1], "Expecting square matrix.")
        ufl_assert(sh[0] is not None, "Unknown dimension.")
        return sh

    def deviatoric(self, o, A):
        sh = self._square_matrix_shape(A)
        if sh[0] == 2:
            return as_matrix([[-1./2*A[1,1]+1./2*A[0,0],A[0,1]],[A[1,0],1./2*A[1,1]-1./2*A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([[-1./3*A[1,1]-1./3*A[2,2]+2./3*A[0,0],A[0,1],A[0,2]],[A[1,0],2./3*A[1,1]-1./3*A[2,2]-1./3*A[0,0],A[1,2]],[A[2,0],A[2,1],-1./3*A[1,1]+2./3*A[2,2]-1./3*A[0,0]]])
        error("dev(A) not implemented for dimension %s." % sh[0])

    def skew(self, o, A):
        i, j = indices(2)
        return as_matrix( (A[i,j] - A[j,i]) / 2, (i,j) )

    def sym(self, o, A):
        i, j = indices(2)
        return as_matrix( (A[i,j] + A[j,i]) / 2, (i,j) )

    def cross(self, o, a, b):
        def c(i, j):
            return Product(a[i],b[j]) - Product(a[j],b[i])
        return as_vector((c(1,2), c(2,0), c(0,1)))

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
        sh = self._square_matrix_shape(A)

        def det2D(B, i, j, k, l):
            return B[i,k]*B[j,l]-B[i,l]*B[j,k]

        if len(sh) == 0:
            return A
        if sh[0] == 2:
            return det2D(A, 0, 1, 0, 1)
        if sh[0] == 3:
            return A[0,0]*det2D(A, 1, 2, 1, 2) + \
                   A[0,1]*det2D(A, 1, 2, 2, 0) + \
                   A[0,2]*det2D(A, 1, 2, 0, 1)
        # TODO: Implement generally for all dimensions?
        error("Determinant not implemented for dimension %d." % self._dim)

    def cofactor(self, o, A):
        sh = self._square_matrix_shape(A)

        if sh[0] == 2:
            return as_matrix([[A[1,1], -A[0,1]], [-A[1,0], A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([ \
                [ A[2,2]*A[1,1] - A[1,2]*A[2,1],
                 -A[0,1]*A[2,2] + A[0,2]*A[2,1],
                  A[0,1]*A[1,2] - A[0,2]*A[1,1]],
                [-A[2,2]*A[1,0] + A[1,2]*A[2,0],
                 -A[0,2]*A[2,0] + A[2,2]*A[0,0],
                  A[0,2]*A[1,0] - A[1,2]*A[0,0]],
                [ A[1,0]*A[2,1] - A[2,0]*A[1,1],
                  A[0,1]*A[2,0] - A[0,0]*A[2,1],
                  A[0,0]*A[1,1] - A[0,1]*A[1,0]] \
                ])
        elif sh[0] == 4:
            # TODO: Find common subexpressions here.
            # TODO: Better implementation?
            return as_matrix([ \
                [-A[3,3]*A[2,1]*A[1,2] + A[1,2]*A[3,1]*A[2,3] + A[1,1]*A[3,3]*A[2,2] - A[3,1]*A[2,2]*A[1,3] + A[2,1]*A[1,3]*A[3,2] - A[1,1]*A[3,2]*A[2,3],
                 -A[3,1]*A[0,2]*A[2,3] + A[0,1]*A[3,2]*A[2,3] - A[0,3]*A[2,1]*A[3,2] + A[3,3]*A[2,1]*A[0,2] - A[3,3]*A[0,1]*A[2,2] + A[0,3]*A[3,1]*A[2,2],
                  A[3,1]*A[1,3]*A[0,2] + A[1,1]*A[0,3]*A[3,2] - A[0,3]*A[1,2]*A[3,1] - A[0,1]*A[1,3]*A[3,2] + A[3,3]*A[1,2]*A[0,1] - A[1,1]*A[3,3]*A[0,2],
                  A[1,1]*A[0,2]*A[2,3] - A[2,1]*A[1,3]*A[0,2] + A[0,3]*A[2,1]*A[1,2] - A[1,2]*A[0,1]*A[2,3] - A[1,1]*A[0,3]*A[2,2] + A[0,1]*A[2,2]*A[1,3]],
                [ A[3,3]*A[1,2]*A[2,0] - A[3,0]*A[1,2]*A[2,3] + A[1,0]*A[3,2]*A[2,3] - A[3,3]*A[1,0]*A[2,2] - A[1,3]*A[3,2]*A[2,0] + A[3,0]*A[2,2]*A[1,3],
                  A[0,3]*A[3,2]*A[2,0] - A[0,3]*A[3,0]*A[2,2] + A[3,3]*A[0,0]*A[2,2] + A[3,0]*A[0,2]*A[2,3] - A[0,0]*A[3,2]*A[2,3] - A[3,3]*A[0,2]*A[2,0],
                 -A[3,3]*A[0,0]*A[1,2] + A[0,0]*A[1,3]*A[3,2] - A[3,0]*A[1,3]*A[0,2] + A[3,3]*A[1,0]*A[0,2] + A[0,3]*A[3,0]*A[1,2] - A[0,3]*A[1,0]*A[3,2],
                  A[0,3]*A[1,0]*A[2,2] + A[1,3]*A[0,2]*A[2,0] - A[0,0]*A[2,2]*A[1,3] - A[0,3]*A[1,2]*A[2,0] + A[0,0]*A[1,2]*A[2,3] - A[1,0]*A[0,2]*A[2,3]],
                [ A[3,1]*A[1,3]*A[2,0] + A[3,3]*A[2,1]*A[1,0] + A[1,1]*A[3,0]*A[2,3] - A[1,0]*A[3,1]*A[2,3] - A[3,0]*A[2,1]*A[1,3] - A[1,1]*A[3,3]*A[2,0],
                  A[3,3]*A[0,1]*A[2,0] - A[3,3]*A[0,0]*A[2,1] - A[0,3]*A[3,1]*A[2,0] - A[3,0]*A[0,1]*A[2,3] + A[0,0]*A[3,1]*A[2,3] + A[0,3]*A[3,0]*A[2,1],
                 -A[0,0]*A[3,1]*A[1,3] + A[0,3]*A[1,0]*A[3,1] - A[3,3]*A[1,0]*A[0,1] + A[1,1]*A[3,3]*A[0,0] - A[1,1]*A[0,3]*A[3,0] + A[3,0]*A[0,1]*A[1,3],
                  A[0,0]*A[2,1]*A[1,3] + A[1,0]*A[0,1]*A[2,3] - A[0,3]*A[2,1]*A[1,0] + A[1,1]*A[0,3]*A[2,0] - A[1,1]*A[0,0]*A[2,3] - A[0,1]*A[1,3]*A[2,0]],
                [-A[1,2]*A[3,1]*A[2,0] - A[2,1]*A[1,0]*A[3,2] + A[3,0]*A[2,1]*A[1,2] - A[1,1]*A[3,0]*A[2,2] + A[1,0]*A[3,1]*A[2,2] + A[1,1]*A[3,2]*A[2,0],
                 -A[3,0]*A[2,1]*A[0,2] - A[0,1]*A[3,2]*A[2,0] + A[3,1]*A[0,2]*A[2,0] - A[0,0]*A[3,1]*A[2,2] + A[3,0]*A[0,1]*A[2,2] + A[0,0]*A[2,1]*A[3,2],
                  A[0,0]*A[1,2]*A[3,1] - A[1,0]*A[3,1]*A[0,2] + A[1,1]*A[3,0]*A[0,2] + A[1,0]*A[0,1]*A[3,2] - A[3,0]*A[1,2]*A[0,1] - A[1,1]*A[0,0]*A[3,2],
                 -A[1,1]*A[0,2]*A[2,0] + A[2,1]*A[1,0]*A[0,2] + A[1,2]*A[0,1]*A[2,0] + A[1,1]*A[0,0]*A[2,2] - A[1,0]*A[0,1]*A[2,2] - A[0,0]*A[2,1]*A[1,2]] \
                ])
        error("Cofactor not implemented for dimension %s." % sh[0])

    def inverse(self, o, A):
        if A.rank() == 0:
            return 1.0 / A
        return self.cofactor(None, A) / self.determinant(None, A)

    # ------------ Compound differential operators

    def div(self, o, a):
        i = Index()
        g = a[i] if a.rank() == 1 else a[...,i]
        return g.dx(i)

    def grad(self, o, a):
        jj = Index()
        if a.rank() > 0:
            ii = tuple(indices(a.rank()))
            return as_tensor(a[ii].dx(jj), ii + (jj,))
        else:
            return as_tensor(a.dx(jj), (jj,))

    def nabla_div(self, o, a):
        i = Index()
        g = a[i] if a.rank() == 1 else a[i,...]
        return g.dx(i)

    def nabla_grad(self, o, a):
        j = Index()
        if a.rank() > 0:
            ii = tuple(indices(a.rank()))
            return as_tensor(a[ii].dx(j), (j,) + ii)
        else:
            return as_tensor(a.dx(j), (j,))

    def curl(self, o, a):
        # o = curl a = "[a.dx(1), -a.dx(0)]"            if a.shape() == ()
        # o = curl a = "cross(nabla, (a0, a1, 0))[2]" if a.shape() == (2,)
        # o = curl a = "cross(nabla, a)"              if a.shape() == (3,)
        def c(i, j):
            return a[j].dx(i) - a[i].dx(j)
        sh = a.shape()
        if sh == ():
            return as_vector((a.dx(1), -a.dx(0)))
        if sh == (2,):
            return c(0,1)
        if sh == (3,):
            return as_vector((c(1,2), c(2,0), c(0,1)))
        error("Invalid shape %s of curl argument." % (sh,))


"""
FIXME: Make expand_compounds2 skip types that we make
work in expand_derivatives, one by one, and optionally
use it instead of expand_compounds from expand_derivatives.
"""

class CompoundExpander2(CompoundExpander):
    def __init__(self, dim):
        CompoundExpander.__init__(self, dim)

    inner = Transformer.reuse_if_possible 
    dot = Transformer.reuse_if_possible


def expand_compounds(e, dim=None):
    """Expand compound objects into basic operators.
    Requires e to have a well defined domain,
    for the geometric dimension to be defined."""
    if dim is None:
        cell = e.cell()
        if cell is not None:
            ufl_assert(not cell.is_undefined(), "Cannot infer dimension from undefined cell.")
            dim = cell.geometric_dimension()
    return apply_transformer(e, CompoundExpander(dim))

def expand_compounds2(e, dim=None):
    """Expand compound objects into basic operators.
    Requires e to have a well defined domain,
    for the geometric dimension to be defined."""
    if dim is None:
        cell = e.cell()
        if cell is not None:
            ufl_assert(not cell.is_undefined(), "Cannot infer dimension from undefined cell.")
            dim = cell.geometric_dimension()
    return apply_transformer(e, CompoundExpander2(dim))
