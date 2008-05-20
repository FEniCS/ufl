"""This module defines the UFLObject base class and all expression
types involved with built-in operators on any UFL object."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-19"

from itertools import chain
from collections import defaultdict
from output import *
#from indexing import *

# This might not all be possible, since UFLObject uses many of these and they in turn inherit UFLObject:
# FIXME: Move all differentiation to differentiation.py
# FIXME: Move all indexing to indexing.py
# FIXME: Move all operators to operators.py
# However, a solution that could work is to attach operator functions UFLObject.__foo__ in the respective foostuff.py where the type __foo__ returns is defined.

#--- Helper functions ---

def is_python_scalar(o):
    return isinstance(o, (int, float))

def is_scalar(o):
    """Return True iff expression is scalar-valued, possibly containing free indices"""
    ufl_assert(isinstance(o, UFLObject), "Assuming an UFLObject.")
    return o.rank() == 0

def is_true_scalar(o):
    """Return True iff expression a single scalar value, with no free indices"""
    return is_scalar(o) and len(o.free_indices()) == 0

class UFLObject(object):
    """Base class of all UFL objects"""
    
    # Freeze member variables (there are none) for objects of this class
    __slots__ = tuple()
    
    #--- Abstract functions that must be implemented by subclasses ---
    
    # All UFL objects must implement operands
    def operands(self):
        "Return a sequence with all subtree nodes in expression tree."
        raise NotImplementedError(self.__class__.operands)
    
    # All UFL objects must implement free_indices
    def free_indices(self):
        "Return a tuple with the free indices (unassigned) of the expression."
        raise NotImplementedError(self.__class__.free_indices)
    
    # All UFL objects must implement rank
    def rank(self):
        """Return the tensor rank of the expression."""
        raise NotImplementedError(self.__class__.rank)
    
    # All UFL objects must implement __repr__
    def __repr__(self):
        """Return string representation of objects"""
        raise NotImplementedError(self.__class__.__repr__)
    
    # All UFL objects must implement __str__
    def __str__(self):
        """Return pretty print string representation of objects"""
        raise NotImplementedError(self.__class__.__str__)
    
    #--- Special functions used for processing expressions ---
    
    def __hash__(self):
        return repr(self).__hash__()
    
    def __eq__(self, other):
        "Checks whether the two expressions are represented the exact same way using repr."
        return repr(self) == repr(other)
    
    # TODO: Keep or remove this? "a in b" is probably ambiguous in may ways.
    def __contains__(self, item):
        """Return whether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        if isinstance(item, UFLObject):
            if item is self:
                return True
            item = repr(item)
        if repr(self) == item:
            return True
        return any((item in o) for o in self.operands())
    
    #--- Basic algebraic operators ---
    
#    def __add__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Sum(self, o)
#    
#    def __radd__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Sum(o, self)
#    
#    def __sub__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return self + (-o)
#    
#    def __rsub__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return o + (-self)
#
#    def __mul__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Product(self, o)
#    
#    def __rmul__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Product(o, self)
#    
#    def __div__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Division(self, o)
#    
#    def __rdiv__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Division(o, self)
#    
#    def __pow__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Power(self, o)
#    
#    def __rpow__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Power(o, self)
#    
#    def __mod__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Mod(self, o)
#    
#    def __rmod__(self, o):
#        if is_python_scalar(o): o = Number(o)
#        if not isinstance(o, UFLObject): return NotImplemented
#        return Mod(o, self)
#    
#    def __neg__(self):
#        return -1*self
#    
#    def __abs__(self):
#        return Abs(self)

    #def _transpose(self):
    #    return Transpose(self)
    
    #T = property(_transpose)

    #--- Indexing ---

    #def __getitem__(self, key):
    #    return Indexed(self, key)
    
    #--- Differentiation ---
    
    #def dx(self, *i):
    #    """Return the partial derivative with respect to spatial variable number i"""
    #    return PartialDerivative(self, i)

#--- Basic terminal objects ---

class Terminal(UFLObject):
    "A terminal node in the UFL expression tree."
    __slots__ = ()
    
    def operands(self):
        "A Terminal object never has operands."
        return tuple()

class Number(Terminal):
    "A constant scalar numeric value."
    __slots__ = ("_value",)
    
    def __init__(self, value):
        self._value = value
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return str(self._value)
    
    def __repr__(self):
        return "Number(%s)" % repr(self._value)

class Symbol(Terminal):
    "A scalar symbol."
    __slots__ = tuple("_name")
    
    def __init__(self, name):
        self._name = str(name)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return self._name
    
    def __repr__(self):
        return "Symbol(%s)" % repr(self._name)


# FIXME: Should we allow a single Variable to represent a tensor expression?
# FIXME: Should a Variable be a Terminal, in which case it doesn't need a symbol but can use a string label,
#        or should it be an UFLObject, in which case it must use Symbol, and we must consider what to do with tensors of Symbol and Variables.
class Variable(UFLObject): # (Terminal):
    __slots__ = ("_symbol", "_expression")
    
    def __init__(self, symbol, expression):
        self._symbol     = symbol if isinstance(symbol, Symbol) else Symbol(symbol)
        self._expression = expression
    
    def operands(self):
        return (self._symbol, self._expression)
    
    def free_indices(self):
        return self._expression.free_indices()
    
    def rank(self):
        return self._expression.rank()
    
    def __str__(self):
        # NB! Doesn't print expression. Is this ok?
        # str shouldn't be used in algorithms, so this is only a matter of choice.
        return str(self._symbol)
    
    def __repr__(self):
        return "Variable(%s, %s)" % (repr(self._symbol), repr(self._expression))

