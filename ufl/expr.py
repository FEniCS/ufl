"""This module defines the Expr class, the superclass 
for all expression tree node types in UFL.

NB! A note about other operators not implemented here:

More operators (special functions) on Exprs are defined in exproperators.py,
as well as the transpose "A.T" and spatial derivative "a.dx(i)".
This is to avoid circular dependencies between Expr and its subclasses.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2011-04-12"

# Modified by Anders Logg, 2008

#--- The base object for all UFL expression tree nodes ---

from collections import defaultdict
from itertools import izip
from ufl.log import warning, error
_class_usage_statistics = defaultdict(int)

def typetuple(e):
    return tuple(type(o) for o in e.operands())

def compute_hash(expr):
    tt = tuple((type(o), typetuple(o)) for o in expr.operands())
    return hash((type(expr), tt))

class Expr(object):
    "Base class for all UFL objects."
    # Freeze member variables for objects of this class
    __slots__ = ()
    #__slots__ = ("_operands", "_hash", "_str", "_repr", "_shape", "_free_indices", "_index_dimensions")
    
    def __init__(self):
        # Comment out this line to disable class construction statistics (used in some unit tests)
        _class_usage_statistics[self.__class__._uflclass] += 1
        #self._hash = None
    
    #=== Abstract functions that must be implemented by subclasses ===
    
    #--- Functions for reconstructing expression ---
    
    # All subclasses must implement reconstruct
    def reconstruct(self, *operands):
        "Return a new object of the same type with new operands."
        raise NotImplementedError(self.__class__.reconstruct)
    
    #--- Functions for expression tree traversal ---
    
    # All subclasses must implement operands
    def operands(self):
        "Return a sequence with all subtree nodes in expression tree."
        #return self._operands
        raise NotImplementedError(self.__class__.operands)
    
    #--- Functions for general properties of expression ---
    
    # All subclasses must implement shape
    def shape(self):
        "Return the tensor shape of the expression."
        #return self._shape
        raise NotImplementedError(self.__class__.shape)
    
    # Subclasses can implement rank if it is known directly (TODO: Is this used anywhere? Usually want to compare shapes anyway.)
    def rank(self):
        "Return the tensor rank of the expression."
        return len(self.shape())
    
    # All subclasses must implement cell if it is known
    def cell(self):
        "Return the cell this expression is defined on."
        for o in self.operands():
            d = o.cell()
            if d is not None:
                return d
        return None
    
    #--- Functions for float evaluation ---
    
    def evaluate(self, x, mapping, component, index_values):
        """Evaluate expression at given coordinate with given values for terminals."""
        raise NotImplementedError(self.__class__.evaluate)
    
    #--- Functions for index handling ---
    
    # All subclasses that can have indices must implement free_indices
    def free_indices(self):
        "Return a tuple with the free indices (unassigned) of the expression."
        #return self._free_indices
        raise NotImplementedError(self.__class__.free_indices)
    
    # All subclasses must implement index_dimensions
    def index_dimensions(self):
        """Return a dict with the free or repeated indices in the expression
        as keys and the dimensions of those indices as values."""
        #return self._index_dimensions
        raise NotImplementedError(self.__class__.index_dimensions)
    
    #--- Special functions for string representations ---
    
    # All subclasses must implement __repr__
    def __repr__(self):
        "Return string representation this object can be reconstructed from."
        #return self._repr
        raise NotImplementedError(self.__class__.__repr__)
    
    # All subclasses must implement __str__
    def __str__(self):
        "Return pretty print string representation of this object."
        #return self._str
        raise NotImplementedError(self.__class__.__str__)
    
    #--- Special functions used for processing expressions ---
    
    def __hash__(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        # Using hash cache to avoid recomputation
        #if self._hash is None:
        
        h = compute_hash(self)
        return h

        #self._hash = h
        #return self._hash
        #return hash(repr(self))
    
    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way using repr. This does not check if the forms
        are mathematically equal or equivalent! Used by sets and dicts."""
        #return (type(self) == type(other)) and ((self is other) or (self.operands() == other.operands()))
        return repr(self) == repr(other)

    def __nonzero__(self):
        "By default, all Expr are nonzero."
        return True 
    
    def __len__(self):
        "Length of expression. Used for iteration over vector expressions."
        s = self.shape()
        if len(s) == 1:
            return s[0]
        error("Cannot take length of non-vector expression.")
    
    def __iter__(self):
        "Iteration over vector expressions."
        for i in range(len(self)):
            yield self[i]
 
    def __floordiv__(self, other):
        "UFL does not support integer division."
        raise NotImplementedError(self.__class__.__floordiv__)

    #def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
    #    "Used for pickle and copy operations."
    #    return self.operands()

#--- Subtypes of Expr, used to logically group class hierarchy ---

class Operator(Expr):
    def __init__(self):
        Expr.__init__(self)
    
    def reconstruct(self, *operands):
        "Return a new object of the same type with new operands."
        return self.__class__._uflclass(*operands)

class WrapperType(Operator):
    def __init__(self):
        Operator.__init__(self)

class AlgebraOperator(Operator):
    def __init__(self):
        Operator.__init__(self)

