"""This module defines the Expr class, the superclass
for all expression tree node types in UFL.

NB! A note about other operators not implemented here:

More operators (special functions) on Exprs are defined in exproperators.py,
as well as the transpose "A.T" and spatial derivative "a.dx(i)".
This is to avoid circular dependencies between Expr and its subclasses.
"""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Modified by Anders Logg, 2008
#
# First added:  2008-03-14
# Last changed: 2012-03-20

#--- The base object for all UFL expression tree nodes ---

from collections import defaultdict
from ufl.log import warning, error

def print_expr_statistics():
    for k in sorted(Expr._class_usage_statistics.keys()):
        born = Expr._class_usage_statistics[k]
        live = born - Expr._class_del_statistics.get(k, 0)
        print "%40s:  %10d  /  %10d" % (k.__name__, live, born)

class Expr(object):
    "Base class for all UFL objects."
    # Freeze member variables for objects of this class
    __slots__ = ()

    _class_usage_statistics = defaultdict(int)
    _class_del_statistics = defaultdict(int)

    def __init__(self):
        # Comment out this line to disable class construction
        # statistics (used in some unit tests)
        Expr._class_usage_statistics[self.__class__._uflclass] += 1

    def x__del__(self): # Enable for profiling
        # Comment out this line to disable class construction
        # statistics (used for manual memory profiling)
        Expr._class_del_statistics[self.__class__._uflclass] += 1

    #=== Abstract functions that must be implemented by subclasses ===

    #--- Functions for reconstructing expression ---

    # All subclasses must implement reconstruct
    def reconstruct(self, *operands):
        "Return a new object of the same type with new operands."
        raise NotImplementedError(self.__class__.reconstruct)

    #--- Functions for expression tree traversal ---

    # All subclasses must implement operands
    def operands(self):
        "Return a sequence with all subtree nodes in expression tree."
        raise NotImplementedError(self.__class__.operands)

    #--- Functions for general properties of expression ---

    # All subclasses must implement shape
    def shape(self):
        "Return the tensor shape of the expression."
        raise NotImplementedError(self.__class__.shape)

    # Subclasses can implement rank if it is known directly
    def rank(self):
        "Return the tensor rank of the expression."
        return len(self.shape())

    # All subclasses must implement domain if it is known
    def domain(self): # TODO: Is it better to use an external traversal algorithm for this?
        "Return the domain this expression is defined on."
        result = None
        for o in self.operands():
            domain = o.domain().top_domain()
            if domain is not None:
                result = domain # Best we have so far
                cell = domain.cell()
                if cell is not None:
                    # A domain with a fully defined cell, we have a winner!
                    break
        return result

    # All subclasses must implement cell if it is known
    def cell(self): # TODO: Deprecate this
        "Return the cell this expression is defined on."
        for o in self.operands():
            cell = o.cell()
            if cell is not None:
                return cell
        return None

    # This function was introduced to clarify and
    # eventually reduce direct dependencies on cells.
    def geometric_dimension(self): # TODO: Deprecate this, use external analysis algorithm
        "Return the geometric dimension this expression lives in."
        cell = self.cell()
        if cell is None:
            error("Cannot get geometric dimension from an expression with no cell!")
        return cell.geometric_dimension()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        raise NotImplementedError(self.__class__.is_cellwise_constant)

    #--- Functions for float evaluation ---

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        """Evaluate expression at given coordinate with given values for terminals."""
        raise NotImplementedError(self.__class__.evaluate)

    def __float__(self):
        if self.shape() != () or self.free_indices() != ():
            raise NotImplementedError(self.__class__.__float__)
        return self(()) # No known x

    #--- Functions for index handling ---

    # All subclasses that can have indices must implement free_indices
    def free_indices(self):
        "Return a tuple with the free indices (unassigned) of the expression."
        raise NotImplementedError(self.__class__.free_indices)

    # All subclasses must implement index_dimensions
    def index_dimensions(self):
        """Return a dict with the free or repeated indices in the expression
        as keys and the dimensions of those indices as values."""
        raise NotImplementedError(self.__class__.index_dimensions)

    #--- Special functions for string representations ---

    # All subclasses must implement signature_data
    def signature_data(self):
        "Return data that uniquely identifies this object."
        raise NotImplementedError(self.__class__.signature_data)

    # All subclasses must implement __repr__
    def __repr__(self):
        "Return string representation this object can be reconstructed from."
        raise NotImplementedError(self.__class__.__repr__)

    # All subclasses must implement __str__
    def __str__(self):
        "Return pretty print string representation of this object."
        raise NotImplementedError(self.__class__.__str__)

    def _repr_latex_(self):
        from ufl.algorithms import ufl2latex
        return "$%s$" % ufl2latex(self)

    def _repr_png_(self):
        from IPython.lib.latextools import latex_to_png
        return latex_to_png(self._repr_latex_())

    #--- Special functions used for processing expressions ---

    def __hash__(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        raise NotImplementedError(self.__class__.__hash__)

    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way. This does not check if the expressions are
        mathematically equal or equivalent! Used by sets and dicts."""
        raise NotImplementedError(self.__class__.__eq__)

    def __nonzero__(self):
        "By default, all Expr are nonzero."
        return True

    def __len__(self):
        "Length of expression. Used for iteration over vector expressions."
        s = self.shape()
        if len(s) == 1:
            return s[0]
        raise NotImplementedError("Cannot take length of non-vector expression.")

    def __iter__(self):
        "Iteration over vector expressions."
        for i in range(len(self)):
            yield self[i]

    def __floordiv__(self, other):
        "UFL does not support integer division."
        raise NotImplementedError(self.__class__.__floordiv__)

    def __pos__(self):
        "Unary + is a no-op."
        return self

    #def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
    #    "Used for pickle and copy operations."
    #    return self.operands()
