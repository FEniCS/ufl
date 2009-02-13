"""This module defines the Expr class, the superclass 
for all expression tree node types in UFL.

NB! A note about other operators not implemented here:

More operators (special functions) on Exprs are defined in exproperators.py,
as well as the transpose "A.T" and spatial derivative "a.dx(i)".
This is to avoid circular dependencies between Expr and its subclasses.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-03"

# Modified by Anders Logg, 2008

#--- The base object for all UFL expression tree nodes ---

from collections import defaultdict
from ufl.log import warning
_class_usage_statistics = defaultdict(int)

class Expr(object):
    "Base class for all UFL objects."
    # Freeze member variables for objects of this class
    __slots__ = ()#"_operands", "_hash")
    
    def __init__(self):
        # Comment out this line to disable class construction statistics (used in some unit tests)
        _class_usage_statistics[self._uflclass] += 1
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
        raise NotImplementedError(self.__class__.operands)
        #return self._operands
    
    #--- Functions for general properties of expression ---
    
    # All subclasses must implement shape
    def shape(self):
        "Return the tensor shape of the expression."
        raise NotImplementedError(self.__class__.shape)
        #return self._shape
    
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
        return ()
        #return self._free_indices
    
    # Subclasses that can have repeated indices must implement repeated_indices
    def repeated_indices(self):
        "Return a tuple with the repeated indices of the expression."
        error("repeated_indices shouldn't be called anymore!")
        return ()
    
    # All subclasses must implement index_dimensions
    def index_dimensions(self):
        """Return a dict with the free or repeated indices in the expression
        as keys and the dimensions of those indices as values."""
        raise NotImplementedError(self.__class__.index_dimensions)
        #return {} # TODO: Might make this optional to implement?
        #return self._index_dimensions
    
    #--- Special functions for string representations ---
    
    # All subclasses must implement __repr__
    def __repr__(self):
        "Return string representation this object can be reconstructed from."
        raise NotImplementedError(self.__class__.__repr__)
        #return self._repr
    
    # All subclasses must implement __str__
    def __str__(self):
        "Return pretty print string representation of this object."
        raise NotImplementedError(self.__class__.__str__)
        #return self._str
    
    #--- Special functions used for processing expressions ---
    
    def __hash__(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        # Using hash cache to avoid recomputation
        #if self._hash is None:
        
        def typetuple(expr):
            return tuple(type(o) for o in expr.operands())
        tt = tuple((type(o), typetuple(o)) for o in self.operands())
        h = hash((type(self), tt))
        return h

        #self._hash = h
        #return self._hash
        #return hash(repr(self))
    
    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way using repr. This does not check if the forms
        are mathematically equal or equivalent! Used by sets and dicts."""
        if type(self) != type(other):
            return False
        if id(self) == other:
            return True
        return self.operands() == other.operands()
        #return (type(self) != type(other)) and ((id(self) == other) or (self.operands() == other.operands()))
        #return repr(self) == repr(other)
    
    def __nonzero__(self):
        "By default, all Expr are nonzero."
        return True 
    
    def __iter__(self):
        raise NotImplementedError
    
    #def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
    #    "Used for pickle and copy operations."
    #    return self.operands()

#--- Subtypes of Expr, used to logically group class hierarchy ---

class Operator(Expr):
    def __init__(self):
        Expr.__init__(self)
    
    def reconstruct(self, *operands):
        "Return a new object of the same type with new operands."
        return self._uflclass(*operands)

class WrapperType(Operator):
    def __init__(self):
        Operator.__init__(self)

class AlgebraOperator(Operator):
    def __init__(self):
        Operator.__init__(self)

