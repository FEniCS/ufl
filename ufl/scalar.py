"This module defines the ScalarValue, IntValue and FloatValue classes."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-01 -- 2008-11-01"

from .output import ufl_assert
from .base import Expr, Terminal

#--- "Low level" scalar types ---

int_type = int
float_type = float
_python_scalar_types = (int_type, float_type)

# TODO: Use high precision float from numpy?
#try:
#    import numpy
#    int_type = numpy.int64
#    float_type = numpy.float96
#    _python_scalar_types += (int_type, float_type)
#except:
#    pass

#--- ScalarValue, FloatValue and IntValue types ---

class ScalarValue(Terminal):
    "A constant scalar value."
    def shape(self):
        return ()
    
    def __eq__(self, other):
        "Allow comparison with python scalars."
        if isinstance(other, ScalarValue):
            return self._value == other._value
        if is_python_scalar(other):
            return self._value == other
        return False
    
    def __str__(self):
        return str(self._value)

class FloatValue(ScalarValue):
    "A constant scalar numeric value."
    __slots__ = ("_value",)
    
    def __new__(cls, value):
        ufl_assert(is_python_scalar(value), "Expecting Python scalar.")
        if value == 0: return Zero()
        return ScalarValue.__new__(cls, value)
    
    def __init__(self, value):
        self._value = float_type(value)
    
    def __repr__(self):
        return "FloatValue(%s)" % repr(self._value)
    
    def __neg__(self):
        return FloatValue(-self._value)

    def __abs__(self):
        return FloatValue(abs(self._value))

class IntValue(ScalarValue):
    "A constant scalar integer value."
    __slots__ = ("_value",)
    
    def __new__(cls, value):
        ufl_assert(is_python_scalar(value), "Expecting Python scalar.")
        if value == 0: return Zero()
        return ScalarValue.__new__(cls, value)
    
    def __init__(self, value):
        self._value = int_type(value)
    
    def __repr__(self):
        return "IntValue(%s)" % repr(self._value)
    
    def __neg__(self):
        return IntValue(-self._value)

    def __abs__(self):
        return IntValue(abs(self._value))

#--- Basic helper functions ---

def is_python_scalar(expression):
    "Return True iff expression is of a Python scalar type."
    return isinstance(expression, _python_scalar_types)

def is_ufl_scalar(expression):
    "Return True iff expression is scalar-valued, but possibly containing free indices."
    return isinstance(expression, Expr) and not expression.shape()

def is_true_ufl_scalar(expression):
    "Return True iff expression is scalar-valued, with no free indices."
    return isinstance(o, Expr) and not (expression.shape() or expression.free_indices())

def as_ufl(expression):
    "Converts expression to an Expr if possible."
    if isinstance(expression, Expr):
        return expression
    if isinstance(expression, int):  
        return IntValue(expression)
    if isinstance(expression, float):  
        return FloatValue(expression)
    ufl_error("Expecting a Python scalar or Expr instance.")
