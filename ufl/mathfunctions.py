"""This module provides basic mathematical functions."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-29"

# Modified by Anders Logg, 2008

from .output import ufl_assert
from .base import Expr, FloatValue, is_true_scalar, is_python_scalar, as_ufl

#--- Function representations ---

class MathFunction(Expr):
    "Base class for all math functions"
    # Freeze member variables for objects in this class
    __slots__ = ("_name", "_argument")
    def __init__(self, name, argument):
        ufl_assert(is_true_scalar(argument), "Expecting scalar argument.")
        self._name     = name
        self._argument = argument
    
    def operands(self):
        return (self._argument,)
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "%s(%s)" % (self._name, self._argument)
    
    def __repr__(self):
        return "%s(%r)" % (self._name, self._argument)

class Sqrt(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "sqrt", argument)

class Exp(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "exp", argument)

class Ln(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "ln", argument)

class Cos(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "cos", argument)

class Sin(MathFunction):
    def __init__(self, argument):
        MathFunction.__init__(self, "sin", argument)

