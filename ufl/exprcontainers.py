"""This module defines special types for representing mapping of expressions to expressions."""

# Copyright (C) 2014 Martin Sandve Alnes
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
from ufl.expr import Expr
from ufl.operatorbase import Operator, WrapperType
from ufl.core.ufl_type import ufl_type

#--- Non-tensor types ---

@ufl_type(num_ops="variable")
class ExprList(WrapperType):
    "List of Expr objects. For internal use, never to be created by end users."
    __slots__ = ("_operands",)
    def __init__(self, *operands):
        WrapperType.__init__(self)
        if not all(isinstance(i, Expr) for i in operands):
            error("Expecting Expr in ExprList.")
        self._operands = operands

    def operands(self):
        return self._operands

    def __getitem__(self, i):
        return self._operands[i]

    def __len__(self):
        return len(self._operands)

    def __iter__(self):
        return iter(self._operands)

    def __str__(self):
        return "ExprList(*(%s,))" % ", ".join(str(i) for i in self._operands)

    def __repr__(self):
        return "ExprList(*%r)" % (self._operands,)

@ufl_type(num_ops="variable")
class ExprMapping(WrapperType):
    "Mapping of Expr objects. For internal use, never to be created by end users."
    __slots__ = ("_operands",)
    def __init__(self, *operands):
        WrapperType.__init__(self)
        if not all(isinstance(e, Expr) for e in operands):
            error("Expecting Expr in ExprMapping.")
        self._operands = operands

    def operands(self):
        return self._operands

    def domains(self):
        # Because this type can act like a terminal if it has no operands, we need to override some recursive operations
        if self._operands:
            return WrapperType.domains()
        else:
            return []

    #def __getitem__(self, key):
    #    return self._operands[key]

    #def __len__(self):
    #    return len(self._operands) // 2

    #def __iter__(self):
    #    return iter(self._operands[::2])

    def __str__(self):
        return "ExprMapping(*%r)" % (self._operands,)

    def __repr__(self):
        return "ExprMapping(*%r)" % (self._operands,)
