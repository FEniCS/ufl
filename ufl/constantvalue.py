"This module defines classes representing constant values."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-01 -- 2009-03-26"

from ufl.log import warning, error
from ufl.assertions import ufl_assert, expecting_python_scalar
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.indexing import Index, FixedIndex

#--- "Low level" scalar types ---

# TODO: Using high precision float from numpy if available?
int_type = int
float_type = float
python_scalar_types = (int, float)
#try:
#    import numpy
#    int_type = numpy.int64
#    float_type = numpy.float96
#    python_scalar_types += (int_type, float_type)
#except:
#    pass

#--- Base classes for constant types ---

class ConstantValue(Terminal):
    def __init__(self):
        Terminal.__init__(self)

class IndexAnnotated(object):
    """Class to annotate expressions that don't depend on
    indices with a set of free indices, used internally to keep
    index properties intact during automatic differentiation."""
    #__slots__ = ("_shape", "_free_indices", "_index_dimensions")
    
    def __init__(self, shape=(), free_indices=(), index_dimensions=None):
        #ufl_assert(all(isinstance(i, int) for i in shape),
        #           "Expecting tuple of int.")
        #ufl_assert(all(isinstance(i, Index) for i in free_indices),
        #           "Expecting tuple of Index objects.")
        all(isinstance(i, int) for i in shape)\
            or error("Expecting tuple of int.")
        all(isinstance(i, Index) for i in free_indices)\
            or error("Expecting tuple of Index objects.")
        self._shape = shape
        self._free_indices = free_indices
        self._index_dimensions = dict(index_dimensions or {})
        #ufl_assert(not (set(self._free_indices) ^ set(self._index_dimensions.keys())),
        #           "Index set mismatch.")
        not (set(self._free_indices) ^ set(self._index_dimensions.keys()))\
            or error("Index set mismatch.")

#--- Class for representing zero tensors of different shapes ---

class Zero(ConstantValue, IndexAnnotated):
    __slots__ = ()
    
    def __init__(self, shape=(), free_indices=(), index_dimensions=None):
        ConstantValue.__init__(self)
        IndexAnnotated.__init__(self, shape, free_indices, index_dimensions)
    
    def reconstruct(self, free_indices=None):
        if not free_indices:
            return self
        ufl_assert(len(free_indices) == len(self._free_indices), "Size mismatch between old and new indices.")
        new_index_dimensions = dict((b, self._index_dimensions[a]) for (a,b) in zip(self._free_indices, free_indices))
        return Zero(self._shape, free_indices, new_index_dimensions)
    
    def shape(self):
        return self._shape
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def evaluate(self, x, mapping, component, index_values):    
        return 0.0
    
    def __str__(self):
        if self._shape == () and self._free_indices == ():
            return "0"
        return "[Zero tensor with shape %s and free indices %s]" % \
            (repr(self._shape), repr(self._free_indices))
    
    def __repr__(self):
        return "Zero(%s, %s, %s)" % (repr(self._shape),
            repr(self._free_indices), repr(self._index_dimensions))
    
    def __eq__(self, other):
        # zero is zero no matter which free indices you look at
        if self._shape == () and other == 0:
            return True
        return isinstance(other, Zero) and self._shape == other.shape()
    
    def __neg__(self):
        return self
    
    def __abs__(self):
        return self
    
    def __nonzero__(self):
        return False 

#--- Scalar value types ---

class ScalarValue(ConstantValue, IndexAnnotated):
    "A constant scalar value."
    __slots__ = ("_value",)
    
    def __new__(cls, value, shape=(), free_indices=(), index_dimensions=None):
        is_python_scalar(value) or expecting_python_scalar(value)
        if value == 0:
            return Zero(shape, free_indices, index_dimensions)
        return ConstantValue.__new__(cls)

    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        ConstantValue.__init__(self)
        IndexAnnotated.__init__(self, shape, free_indices, index_dimensions)
        self._value = value
    
    def reconstruct(self, free_indices=None):
        "Reconstruct with new free indices."
        if not free_indices:
            return self
        ufl_assert(len(free_indices) == len(self._free_indices), "Size mismatch between old and new indices.")
        new_index_dimensions = dict((b, self._index_dimensions[a]) for (a,b) in zip(self._free_indices, free_indices))
        return self._uflclass(self._value, self._shape, free_indices, new_index_dimensions)
    
    def shape(self):
        return self._shape
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def value(self):
        return self._value
    
    def evaluate(self, x, mapping, component, index_values):
        return self._value
    
    def __eq__(self, other):
        "This is implemented to allow comparison with python scalars."
        return self._value == other
    
    def __str__(self):
        return str(self._value)
    
    def __float__(self):
        return float(self._value)
    
    def __int__(self):
        return int(self._value)
    
    def __neg__(self):
        return type(self)(-self._value)

    def __abs__(self):
        return type(self)(abs(self._value))

class FloatValue(ScalarValue):
    "A constant scalar numeric value."
    __slots__ = ()
    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        ScalarValue.__init__(self, float_type(value), shape, free_indices, index_dimensions)
        self._repr = "%s(%s, %s, %s, %s)" % (type(self).__name__, repr(self._value), repr(self._shape), repr(self._free_indices), repr(self._index_dimensions))
    
    def __repr__(self):
        return self._repr
    
class IntValue(ScalarValue):
    "A constant scalar integer value."
    __slots__ = ()
    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        ScalarValue.__init__(self, int_type(value), shape, free_indices, index_dimensions)
        self._repr = "%s(%s, %s, %s, %s)" % (type(self).__name__, repr(self._value), repr(self._shape), repr(self._free_indices), repr(self._index_dimensions))

    def __repr__(self):
        return self._repr
    
class ScalarSomething(ScalarValue):
    """A scalar value of some externally defined type.
    
    Using this will likely break many algorithms, in particular
    AD can't possibly know what to do with it."""
    __slots__ = ()
    def __init__(self, value):
        ScalarValue.__init__(self, value)
        self._repr = "ScalarSomething(%s)" % repr(self._value)
    
    def evaluate(self, x, mapping, component, index_values):
        return float(self)
    
    def __repr__(self):
        return self._repr

#--- Identity matrix ---

class Identity(ConstantValue):
    __slots__ = ("_dim", "_repr")

    def __init__(self, dim):
        ConstantValue.__init__(self)
        self._dim = dim
        self._repr = "Identity(%d)" % self._dim
    
    def shape(self):
        return (self._dim, self._dim)
    
    def evaluate(self, x, mapping, component, index_values):
        a, b = component
        return 1 if a == b else 0
    
    def __getitem__(self, key):
        if isinstance(key[0], (int, FixedIndex)) and isinstance(key[1], (int, FixedIndex)):
            return IntValue(1) if (int(key[0]) == int(key[1])) else Zero()
        return Expr.__getitem__(self, key)
    
    def __str__(self):
        return "I"
    
    def __repr__(self):
        return self._repr
    
    def __eq__(self, other):
        return isinstance(other, Identity) and self._dim == other._dim

#--- Helper functions ---

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
    if isinstance(expression, (int, int_type)):
        return IntValue(expression)
    if isinstance(expression, (float, float_type)):
        return FloatValue(expression)
    warning("Wrapping non-UFL expression. This is experimental and will likely break many algorithms!")
    return ScalarSomething(expression)

