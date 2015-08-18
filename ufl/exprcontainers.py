# -*- coding: utf-8 -*-
"""This module defines special types for representing mapping of expressions to expressions."""

# Copyright (C) 2014 Martin Sandve Alnæs
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

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import EmptyDict
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type

#--- Non-tensor types ---

@ufl_type(num_ops="varying")
class ExprList(Operator):
    "List of Expr objects. For internal use, never to be created by end users."
    __slots__ = ()

    def __init__(self, *operands):
        Operator.__init__(self, operands)
        if not all(isinstance(i, Expr) for i in operands):
            error("Expecting Expr in ExprList.")

    def __getitem__(self, i):
        return self.ufl_operands[i]

    def __len__(self):
        return len(self.ufl_operands)

    def __iter__(self):
        return iter(self.ufl_operands)

    def __str__(self):
        return "ExprList(*(%s,))" % ", ".join(str(i) for i in self.ufl_operands)

    def __repr__(self):
        return "ExprList(*%r)" % (self.ufl_operands,)

    @property
    def ufl_shape(self):
        error("A non-tensor type has no ufl_shape.")

    @property
    def ufl_free_indices(self):
        error("A non-tensor type has no ufl_free_indices.")

    def free_indices(self):
        error("A non-tensor type has no free_indices.")

    @property
    def ufl_index_dimensions(self):
        error("A non-tensor type has no ufl_index_dimensions.")

    def index_dimensions(self):
        error("A non-tensor type has no index_dimensions.")


@ufl_type(num_ops="varying")
class ExprMapping(Operator):
    "Mapping of Expr objects. For internal use, never to be created by end users."
    __slots__ = ()

    def __init__(self, *operands):
        Operator.__init__(self, operands)
        if not all(isinstance(e, Expr) for e in operands):
            error("Expecting Expr in ExprMapping.")

    def domains(self):
        # Because this type can act like a terminal if it has no operands, we need to override some recursive operations
        if self.ufl_operands:
            return Operator.domains()
        else:
            return []

    #def __getitem__(self, key):
    #    return self.ufl_operands[key]

    #def __len__(self):
    #    return len(self.ufl_operands) // 2

    #def __iter__(self):
    #    return iter(self.ufl_operands[::2])

    def __str__(self):
        return "ExprMapping(*%r)" % (self.ufl_operands,)

    def __repr__(self):
        return "ExprMapping(*%r)" % (self.ufl_operands,)

    @property
    def ufl_shape(self):
        error("A non-tensor type has no ufl_shape.")

    @property
    def ufl_free_indices(self):
        error("A non-tensor type has no ufl_free_indices.")

    def free_indices(self):
        error("A non-tensor type has no free_indices.")

    @property
    def ufl_index_dimensions(self):
        error("A non-tensor type has no ufl_index_dimensions.")

    def index_dimensions(self):
        error("A non-tensor type has no index_dimensions.")
