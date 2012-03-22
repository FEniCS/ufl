"""Lifting operations."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# First added:  2009-06-19
# Last changed: 2011-06-02

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.operatorbase import Operator
from ufl.finiteelement import FiniteElementBase
from ufl.operators import jump
from ufl.common import EmptyDict

class LiftingResult(Operator):
    def __init__(self, operator, operand):
        Operator.__init__(self)
        ufl_assert(operand.free_indices() == (), "Not expecting free indices in operand to lifting operator.")
        ufl_assert(isinstance(operator, TerminalOperator), "Expecting a lifting operator.")
        ufl_assert(isinstance(operand, Expr), "Expecting an Expr.")
        self._operator = operator
        self._operand = operand

        # FIXME: Compute shape without making jump so we don't need the cell here:
        cell = self._operator.cell()
        n = cell.n
        self._shape = jump(self._operand, n).shape() # FIXME: Is this right?

    def operands(self):
        return (self._operator, self._operand)
    
    def shape(self):
        return self._shape

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return EmptyDict

    def evaluate(self, x, mapping, component, index_values):
        error("Evaluate can not easily be implemented for this type.")

class LiftingOperatorResult(LiftingResult):
    def __init__(self, operator, operand):
        LiftingResult.__init__(self, operator, operand)

    def __str__(self):
        return "%s(%s)" % (self._operator, self._operand,)

    def __repr__(self):
        return "LiftingOperatorResult(%r, %r)" % (self._operator, self._operand)

class LiftingFunctionResult(LiftingResult):
    def __init__(self, operator, operand):
        LiftingResult.__init__(self, operator, operand)

    def __str__(self):
        return "%s(%s)" % (self._operator, self._operand,)

    def __repr__(self):
        return "LiftingFunctionResult(%r, %r)" % (self._operator, self._operand)

class TerminalOperator(Terminal):
    def __init__(self):
        Terminal.__init__(self)
    
    def shape(self):
        error("Calling this makes no sense.")
        return ()

    def free_indices(self):
        error("Calling this makes no sense.")
        return ()

    def index_dimensions(self):
        error("Calling this makes no sense.")
        return EmptyDict

    def evaluate(self, x, mapping, component, index_values):
        error("Evaluate can not easily be implemented for this type.")

class LiftingOperator(TerminalOperator):
    def __init__(self, element):
        TerminalOperator.__init__(self)
        ufl_assert(isinstance(element, FiniteElementBase), "Expecting a finite element.")
        self._element = element

    def cell(self):
        return self._element.cell()

    def __str__(self):
        return "r[%s]" % self._element.shortstr() # FIXME: Is it R here?

    def __repr__(self):
        return "LiftingOperator(%r)" % (self._element,)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            a, = args
            if isinstance(a, Expr):
                return LiftingOperatorResult(self, a)
        return Terminal.__call__(self, *args, **kwargs)

class LiftingFunction(TerminalOperator):
    def __init__(self, element):
        TerminalOperator.__init__(self)
        ufl_assert(isinstance(element, FiniteElementBase), "Expecting a finite element.")
        self._element = element

    def cell(self):
        return self._element.cell()

    def __str__(self):
        return "R[%s]" % self._element.shortstr() # FIXME: Is it r here?

    def __repr__(self):
        return "LiftingFunction(%r)" % (self._element,)

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            a, = args
            if isinstance(a, Expr):
                return LiftingFunctionResult(self, a)
        return Terminal.__call__(self, *args, **kwargs)

