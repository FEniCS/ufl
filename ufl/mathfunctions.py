"""This module provides basic mathematical functions."""

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
# Modified by Anders Logg, 2008
# Modified by Kristian B. Oelgaard, 2011
#
# First added:  2008-03-14
# Last changed: 2011-10-20

import math
from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.expr import Operator
from ufl.constantvalue import is_true_ufl_scalar, ScalarValue, Zero, FloatValue

"""
TODO: Include additional functions available in <cmath> (need derivatives as well):

Trigonometric functions:
atan2    Compute arc tangent with two parameters (function)

Hyperbolic functions:
cosh     Compute hyperbolic cosine (function)
sinh     Compute hyperbolic sine (function)
tanh     Compute hyperbolic tangent (function)

Exponential and logarithmic functions:
log10    Compute common logarithm (function)

TODO: Include bessel functions, need non-standard library implementation in generated code.

TODO: Other special functions?
"""

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
        try:
            res = getattr(math, self._name)(a)
        except ValueError:
            warning('Value error in evaluation of function %s with argument %s.' % (self._name, a))
            raise
        return res
    
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

class Erf(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.erf(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "erf", argument)
