# -*- coding: utf-8 -*-
"Base class for all operators, i.e. non-terminal expr types."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from six import iteritems

from ufl.log import error
from ufl.core.expr import Expr
from ufl.core.ufl_type import ufl_type
from ufl.core.multiindex import Index


#--- Base class for operator objects ---

@ufl_type(is_abstract=True, is_terminal=False)
class Operator(Expr):
    __slots__ = ("ufl_operands",)

    def __init__(self, operands=None):
        Expr.__init__(self)

        # If operands is None, the type sets this itself. This is to get around
        # some tricky too-fancy __new__/__init__ design in algebra.py, for now.
        # It would be nicer to make the classes in algebra.py pass operands here.
        if operands is not None:
            self.ufl_operands = operands

    def reconstruct(self, *operands):
        "Return a new object of the same type with new operands."
        return self._ufl_class_(*operands)

    def _ufl_signature_data_(self):
        return self._ufl_typecode_

    def _ufl_compute_hash_(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        return hash((self._ufl_typecode_,) + tuple(hash(o) for o in self.ufl_operands))

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return all(o.is_cellwise_constant() for o in self.ufl_operands)

    # --- Transitional property getters ---

    def operands(self):
        "Intermediate helper property getter to transition from .operands() to .ufl_operands."
        deprecate("Expr.operands() is deprecated, please use property Expr.ufl_operands instead.")
        return self.ufl_operands

    def free_indices(self):
        "Intermediate helper property getter to transition from .free_indices() to .ufl_free_indices."
        deprecate("Expr.free_indices() is deprecated, please use property Expr.ufl_free_indices instead.")
        return tuple(Index(count=i) for i in self.ufl_free_indices)

    def index_dimensions(self):
        "Intermediate helper property getter to transition from .index_dimensions() to .ufl_index_dimensions."
        deprecate("Expr.index_dimensions() is deprecated, please use property Expr.ufl_index_dimensions instead.")
        return { Index(count=i): d for i, d in zip(self.ufl_free_indices, self.ufl_index_dimensions) }
