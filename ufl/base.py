"""This module defines the UFLObject base class and all expression
types involved with built-in operators on any UFL object."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-26"

# Modified by Anders Logg, 2008

# UFL imports
from .output import ufl_assert

#--- The base object for all UFL expression tree nodes ---

class UFLObject(object):
    "Base class for all UFL objects."
    
    # Freeze member variables (there are none) for objects of this class
    __slots__ = ()
    
    #--- Abstract functions that must be implemented by subclasses ---
    
    # All subclasses must implement operands
    def operands(self):
        "Return a sequence with all subtree nodes in expression tree."
        raise NotImplementedError(self.__class__.operands)
    
    # All subclasses must implement free_indices
    def free_indices(self):
        "Return a tuple with the free indices (unassigned) of the expression."
        raise NotImplementedError(self.__class__.free_indices)

    # All subclasses must implement repeated_index_dimensions 
    # if they can have any repeated indices
    def repeated_index_dimensions(self, default_dim):
        """Return a dict with the repeated indices in the expression
        as keys and the dimensions of those indices as values."""
        return {}

    # All subclasses must implement free_index_dimensions
    #def free_index_dimensions(self): # FIXME: Add this?
    #    "Return a tuple with the dimensions of the free indices."
    #    raise NotImplementedError(self.__class__.free_index_dimensions)
    
    # All subclasses must implement shape
    def shape(self):
        "Return the tensor shape of the expression."
        raise NotImplementedError(self.__class__.shape)
    
    def rank(self):
        "Return the tensor rank of the expression."
        return len(self.shape())

    # Objects (operators) are linear if not overloaded otherwise by subclass
    def is_linear(self):
        "Return true iff object is linear."
        return True
    
    # All subclasses must implement __repr__
    def __repr__(self):
        "Return string representation this object can be reconstructed from."
        raise NotImplementedError(self.__class__.__repr__)
    
    # All subclasses must implement __str__
    def __str__(self):
        "Return pretty print string representation of this object."
        raise NotImplementedError(self.__class__.__str__)
    
    #--- Special functions used for processing expressions ---
    
    def __hash__(self):
        "Compute a hash code for this expression."
        return hash((type(self), tuple(type(o) for o in self.operands())))
        #return hash(repr(self))
    
    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way using repr. This does not check if the forms
        are mathematically equal or equivalent!"""
        return repr(self) == repr(other)

    def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
        "Used for pickle and copy operations."
        return self.operands()


#--- A note about other operators ---

# More operators (special functions) on UFLObjects are defined in baseoperators.py,
# as well as the transpose "A.T" and spatial derivative "a.dx(i)".

#--- Base class for terminal objects ---

class Terminal(UFLObject):
    "A terminal node in the UFL expression tree."
    __slots__ = ()
    
    def operands(self):
        "A Terminal object never has operands."
        return ()

#--- Zero tensors of different shapes ---

class ZeroType(Terminal): # Experimental!
    __slots__ = ("_shape",)
    def __init__(self, shape):
        self._shape = shape
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "<Zero tensor with shape %s>" % repr(self._shape)
    
    def __repr__(self):
        return "ZeroType(%s)" % repr(self._shape)

_zero_cache = {}
def zero_tensor(shape):
    if shape in _zero_cache:
        return _zero_cache[shape]
    z = ZeroType(shape)
    _zero_cache[shape] = z
    return z

def zero():
    return zero_tensor(())

def is_zero(expression):
    return isinstance(expression, ZeroType) or expression == 0

#--- FloatValue type ---

class FloatValue(Terminal):
    "A constant scalar numeric value."
    __slots__ = ("_value",)
    
    def __init__(self, value):
        self._value = value
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return ()
    
    def __eq__(self, other):
        "Allow comparison with python scalars."
        if is_python_scalar(other):
            return self._value == other
        return UFLObject.__eq__(self, other)
    
    def __str__(self):
        return str(self._value)
    
    def __repr__(self):
        return "FloatValue(%s)" % repr(self._value)

def float_value(value): # TODO: Use metaclass instead to return zero if value is 0?
    if value == 0:
        return zero()
    return FloatValue(value)

#--- Base class of compound objects ---

class Compound(UFLObject):
    "An object that can be also expressed as a combination of simpler operations."
    __slots__ = ()
    def __init__(self):
        UFLObject.__init__(self)
    
    def as_basic(self, *operands):
        "Return this expression expressed using basic operations."
        raise NotImplementedError(self.__class__.as_basic)

#--- Basic helper functions ---

def is_python_scalar(o):
    return isinstance(o, (int, float))

def is_scalar(o):
    "Return True iff expression is scalar-valued, possibly containing free indices."
    ufl_assert(isinstance(o, UFLObject), "Assuming an UFLObject.")
    return o.shape() == ()

def is_true_scalar(o):
    "Return True iff expression a single scalar value, with no free indices."
    return is_scalar(o) and len(o.free_indices()) == 0

def as_ufl(o):
    "Returns o if it is an UFLObject or an UFLObject wrapper if o is a scalar."  
    if is_python_scalar(o):
        return float_value(o)
    ufl_assert(isinstance(o, UFLObject), "Expecting Python scalar or UFLObject instance.")
    return o
    
