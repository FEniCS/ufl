"""This module defines the IndexSum class."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.algebra import Product
from ufl.constantvalue import Zero
from ufl.core.expr import Expr, ufl_err_str
from ufl.core.multiindex import MultiIndex
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.indexed import Indexed
from ufl.precedence import parstr

# --- Sum over an index ---


@ufl_type(num_ops=2)
class IndexSum(Operator):
    """Index sum."""

    __slots__ = ("_dimension", "_initialised", "ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, summand, index):
        """Create a new IndexSum."""
        # Error checks
        if not isinstance(summand, Expr):
            raise ValueError(f"Expecting Expr instance, got {ufl_err_str(summand)}")
        if not isinstance(index, MultiIndex):
            raise ValueError(f"Expecting MultiIndex instance, got {ufl_err_str(index)}")
        if len(index) != 1:
            raise ValueError(f"Expecting a single Index but got {len(index)}.")

        (j,) = index
        # Simplification to zero
        if isinstance(summand, Zero):
            sh = summand.ufl_shape
            fi = summand.ufl_free_indices
            fid = summand.ufl_index_dimensions
            pos = fi.index(j.count())
            fi = fi[:pos] + fi[pos + 1 :]
            fid = fid[:pos] + fid[pos + 1 :]
            return Zero(sh, fi, fid)

        # Factor out common factors
        if isinstance(summand, Product):
            a, b = summand.ufl_operands
            if j.count() not in a.ufl_free_indices:
                return Product(a, IndexSum(b, index))
            elif j.count() not in b.ufl_free_indices:
                return Product(b, IndexSum(a, index))

        self = Operator.__new__(cls)
        self._initialised = False
        return self

    def __init__(self, summand, index):
        """Initialise."""
        if self._initialised:
            return
        (j,) = index
        fi = summand.ufl_free_indices
        fid = summand.ufl_index_dimensions
        pos = fi.index(j.count())
        self._dimension = fid[pos]
        self.ufl_free_indices = fi[:pos] + fi[pos + 1 :]
        self.ufl_index_dimensions = fid[:pos] + fid[pos + 1 :]
        Operator.__init__(self, (summand, index))
        self._initialised = True

    def index(self):
        """Get index."""
        return self.ufl_operands[1][0]

    def dimension(self):
        """Get dimension."""
        return self._dimension

    @property
    def ufl_shape(self):
        """Get UFL shape."""
        return self.ufl_operands[0].ufl_shape

    def _simplify_indexed(self, multiindex):
        """Return a simplified Expr used in the constructor of Indexed(self, multiindex)."""
        A, i = self.ufl_operands
        return IndexSum(Indexed(A, multiindex), i)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        (i,) = self.ufl_operands[1]
        tmp = 0
        for k in range(self._dimension):
            index_values.push(i, k)
            tmp += self.ufl_operands[0].evaluate(x, mapping, component, index_values)
            index_values.pop()
        return tmp

    def __str__(self):
        """Format as a string."""
        return f"sum_{{{self.ufl_operands[1]!s}}} {parstr(self.ufl_operands[0], self)} "
