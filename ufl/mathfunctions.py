"""This module provides basic mathematical functions."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-03-14 -- 2009-04-21"

# Modified by Anders Logg, 2008

import math
from ufl.assertions import ufl_assert
from ufl.expr import Operator
from ufl.constantvalue import is_true_ufl_scalar, ScalarValue, Zero, FloatValue

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
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.sqrt(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "sqrt", argument)

class Exp(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.exp(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "exp", argument)

class Ln(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.log(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "ln", argument)
    
    def evaluate(self, x, mapping, component, index_values):    
        a = self._argument.evaluate(x, mapping, component, index_values)
        return math.log(a)

class Cos(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.cos(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "cos", argument)

class Sin(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.sin(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "sin", argument)

class Tan(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.tan(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "tan", argument)

class Acos(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.acos(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "acos", argument)

class Asin(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.asin(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "asin", argument)

class Atan(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.atan(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "atan", argument)
