"""This module defines the UFLObject base class and all expression
types involved with built-in operators on any UFL object."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-06-08"

# External imports
from itertools import chain
from collections import defaultdict

# UFL imports
from output import *

#--- The base object for all UFL expression tree nodes ---

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
    
#--- A note about other operators ---

# Operators on UFLObjects are defined in other modules as outlined below.
#
# 1. For the definition of UFLObject.T, see tensoralgebra.py.
#
# 2. For the definition of algebraic operators, see algebra.py
#    where the following operators are defined:
#
#    def __add__(self, o):
#    def __radd__(self, o):
#    def __sub__(self, o):
#    def __rsub__(self, o):
#    def __mul__(self, o):
#    def __rmul__(self, o):
#    def __div__(self, o):
#    def __rdiv__(self, o):
#    def __pow__(self, o):
#    def __rpow__(self, o):
#    def __mod__(self, o):
#    def __rmod__(self, o):
#    def __neg__(self):
#    def __abs__(self):
#
# 3. For indexing operations, see indexing.py
#    where the following operator is defined:
#
#    def __getitem__(self, key):
#
# 4. For restriction operations, see restriction.py
#    where the following operator is defined:
#
#    def __call__(self, side):
#
# 5. For differentiation operations, see differentiation.py
#    where the following operator is defined:
#
#    def dx(self, *i):

#--- Basic terminal objects ---

class Terminal(UFLObject):
    "A terminal node in the UFL expression tree"
    __slots__ = ()
    
    def operands(self):
        "A Terminal object never has operands"
        return tuple()

class Number(Terminal):
    "A constant scalar numeric value"
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

#--- Basic helper functions ---

def is_python_scalar(o):
    return isinstance(o, (int, float))

def is_scalar(o):
    """Return True iff expression is scalar-valued, possibly containing free indices"""
    ufl_assert(isinstance(o, UFLObject), "Assuming an UFLObject.")
    return o.rank() == 0

def is_true_scalar(o):
    """Return True iff expression a single scalar value, with no free indices"""
    return is_scalar(o) and len(o.free_indices()) == 0
