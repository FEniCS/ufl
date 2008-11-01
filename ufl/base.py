"""This module defines the Expr base class and all expression
types involved with built-in operators on any UFL object."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-11-01"

# Modified by Anders Logg, 2008

#--- The base object for all UFL expression tree nodes ---

class Expr(object):
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
    
    # TODO: Must all subclasses implement free_index_dimensions?
    def free_index_dimensions(self, default_dim):
        """Return a dict with the free indices in the expression
        as keys and the dimensions of those indices as values."""
        # TODO: Implement this everywhere. Need it to get the right shape of ComponentTensor.
        #raise NotImplementedError(self.__class__.free_index_dimensions)
        # This implementation works for all types as long as the
        # indices aren't indexing something with non-default dimensions...
        # Perhaps we could disallow indexing of non-default dimension sizes?
        return dict((i, default_dim) for i in self.free_indices())
    
    # Subclasses that can have repeated indices
    # must implement repeated_indices
    def repeated_indices(self):
        "Return a tuple with the repeated indices of the expression."
        return ()
    
    # Subclasses that can have repeated indices
    # must implement repeated_index_dimensions
    def repeated_index_dimensions(self, default_dim):
        """Return a dict with the repeated indices in the expression
        as keys and the dimensions of those indices as values."""
        return {}
    
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
        def typetuple(e):
            return tuple(type(o) for o in e.operands())
        tt = tuple((type(o), typetuple(o)) for o in self.operands())
        return hash((type(self), tt))
        #return hash(repr(self))
    
    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way using repr. This does not check if the forms
        are mathematically equal or equivalent!"""
        if type(self) != type(other):
            return False
        if id(self) == other:
            return True
        return self.operands() == other.operands()
        #return repr(self) == repr(other)
    
    def __nonzero__(self):
        "By default, all Expr are nonzero."
        return True 

    def __iter__(self):
        raise NotImplementedError
    
    def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
        "Used for pickle and copy operations."
        return self.operands()

#--- A note about other operators ---

# More operators (special functions) on Exprs are defined in baseoperators.py,
# as well as the transpose "A.T" and spatial derivative "a.dx(i)".
# This is to avoid circular dependencies between Expr and its subclasses.

#--- Base class for terminal objects ---

class Terminal(Expr):
    "A terminal node in the UFL expression tree."
    __slots__ = ()
    
    def operands(self):
        "A Terminal object never has operands."
        return ()
    
    def free_indices(self):
        "A Terminal object never has free indices."
        return ()
    
    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way using repr. This does not check if the forms
        are mathematically equal or equivalent!"""
        if type(self) != type(other):
            return False
        if id(self) == other:
            return True
        return repr(self) == repr(other)
    
    def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
        "Used for pickle and copy operations."
        raise NotImplementedError, "Must reimplement in each Terminal, or?"

