"""Lifting operations."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-06-19 -- 2009-06-19"

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr, Operator
from ufl.terminal import Terminal
from ufl.finiteelement import FiniteElementBase
from ufl.operators import jump

class LiftingResult(Operator):
    def __init__(self, operator, operand):
        Operator.__init__(self)
        ufl_assert(operand.free_indices() == (), "Not expecting free indices in operand to lifting operator.")
        ufl_assert(isinstance(operator, TerminalOperator), "Expecting a lifting operator.")
        ufl_assert(isinstance(operand, Expr), "Expecting an Expr.")
        self._operator = operator
        self._operand = operand
        self._shape = jump(self._operand, self._operator.cell().n).shape() # FIXME: Is this right?

    def operands(self):
        return (self._operator, self._operand)
    
    def shape(self):
        return self._shape

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return {}

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
        return {}

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

