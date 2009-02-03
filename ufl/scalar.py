"This module defines the ScalarValue, IntValue and FloatValue classes."


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-01 -- 2009-02-03"

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.terminal import ConstantValue
from ufl.zero import Zero

#--- "Low level" scalar types ---

int_type = int
float_type = float
python_scalar_types = (int_type, float_type)

# TODO: Use high precision float from numpy?
#try:
#    import numpy
#    int_type = numpy.int64
#    float_type = numpy.float96
#    python_scalar_types += (int_type, float_type)
#except:
#    pass

#--- ScalarValue, FloatValue and IntValue types ---

class ScalarValue(ConstantValue):
    "A constant scalar value."
    __slots__ = ("_value",)
    def __init__(self, value):
        ConstantValue.__init__(self)
        self._value = value
    
    def shape(self):
        return ()
    
    def __str__(self):
        return str(self._value)
    
    def __float__(self):
        return float(self._value)
    
    def __int__(self):
        return int(self._value)

class FloatValue(ScalarValue):
    "A constant scalar numeric value."
    __slots__ = ()
    def __new__(cls, value):
        ufl_assert(is_python_scalar(value), "Expecting Python scalar.")
        if value == 0: return Zero()
        return ScalarValue.__new__(cls, value)
    
    def __init__(self, value):
        ScalarValue.__init__(self, float_type(value))
    
    def evaluate(self, x, mapping, component, index_values):
        return float(self)
    
    def __eq__(self, other):
        "This is implemented to allow comparison with python scalars."
        return self._value == other
    
    def __repr__(self):
        return "FloatValue(%s)" % repr(self._value)
    
    def __neg__(self):
        return FloatValue(-self._value)

    def __abs__(self):
        return FloatValue(abs(self._value))

class IntValue(ScalarValue):
    "A constant scalar integer value."
    __slots__ = ()
    def __new__(cls, value):
        ufl_assert(is_python_scalar(value), "Expecting Python scalar.")
        if value == 0: return Zero()
        return ScalarValue.__new__(cls, value)
    
    def __init__(self, value):
        ScalarValue.__init__(self, int_type(value))
    
    def evaluate(self, x, mapping, component, index_values):
        return int(self)
    
    def __eq__(self, other):
        "This is implemented to allow comparison with python scalars."
        return self._value == other
    
    def __repr__(self):
        return "IntValue(%s)" % repr(self._value)
    
    def __neg__(self):
        return IntValue(-self._value)

    def __abs__(self):
        return IntValue(abs(self._value))

class ScalarSomething(ScalarValue):
    """A scalar value of some externally defined type.
    
    Using this will likely break many algorithms, in particular
    AD can't possibly know what to do with it."""
    __slots__ = ()
    def __init__(self, value):
        ScalarValue.__init__(self, value)
    
    def evaluate(self, x, mapping, component, index_values):
        return float(self)
    
    def __repr__(self):
        return "ScalarSomething(%s)" % repr(self._value)
    
    def __neg__(self):
        return ScalarSomething(-self._value)
    
    def __abs__(self):
        return ScalarSomething(abs(self._value))

#--- Basic helper functions ---

def is_python_scalar(expression):
    "Return True iff expression is of a Python scalar type."
    return isinstance(expression, python_scalar_types)

def is_ufl_scalar(expression):
    """Return True iff expression is scalar-valued,
    but possibly containing free indices."""
    return isinstance(expression, Expr) and not expression.shape()

def is_true_ufl_scalar(expression):
    """Return True iff expression is scalar-valued,
    with no free indices."""
    return isinstance(expression, Expr) and \
        not (expression.shape() or expression.free_indices())

def as_ufl(expression):
    "Converts expression to an Expr if possible."
    if isinstance(expression, Expr):
        return expression
    if isinstance(expression, int):  
        return IntValue(expression)
    if isinstance(expression, float):  
        return FloatValue(expression)
    warning("Wrapping non-UFL expression. This is experimental!")
    return ScalarSomething(expression)
