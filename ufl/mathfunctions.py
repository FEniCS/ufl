"""This module provides basic mathematical functions."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-20"

# Modified by Anders Logg, 2008

import math
from ufl.assertions import ufl_assert
from ufl.expr import Operator
from ufl.constantvalue import FloatValue, is_true_ufl_scalar, is_python_scalar, as_ufl

#--- Function representations ---

class MathFunction(Operator):
    "Base class for all math functions"
    # Freeze member variables for objects in this class
    __slots__ = ("_name", "_argument", "_repr")
    def __init__(self, name, argument):
        Operator.__init__(self)
        ufl_assert(is_true_ufl_scalar(argument), "Expecting scalar argument.")
        self._name     = name
        self._argument = argument
        self._repr = "%s(%r)" % (name, argument)
    
    def operands(self):
        return (self._argument,)
    
    def free_indices(self):
        return ()
    
    def index_dimensions(self):
        return {}
    
    def shape(self):
        return ()
    
    def evaluate(self, x, mapping, component, index_values):    
        a = self._argument.evaluate(x, mapping, component, index_values)
        return getattr(math, self._name)(a)
    
    def __str__(self):
        return "%s(%s)" % (self._name, self._argument)
    
    def __repr__(self):
        return self._repr

class Sqrt(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "sqrt", argument)

class Exp(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "exp", argument)

class Ln(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "ln", argument)
    
    def evaluate(self, x, mapping, component, index_values):    
        a = self._argument.evaluate(x, mapping, component, index_values)
        return math.log(a)

class Cos(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "cos", argument)

class Sin(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "sin", argument)

