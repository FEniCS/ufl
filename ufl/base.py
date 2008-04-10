"""This module defines the UFLObject base class and all expression
types involved with built-in operators on any UFL object."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-04-02"

import operator
from itertools import chain
from collections import defaultdict
from output import *
from indexing import Index, MultiIndex, extract_indices

# FIXME: Move all differentiation to differentiation.py
# FIXME: Move all indexing to indexing.py
# FIXME: Move all operators to operators.py

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
    
    #--- Basic algebraic operators ---
    
    def __add__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Sum(self, o)
    
    def __radd__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Sum(o, self)
    
    def __sub__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return self + (-o)
    
    def __rsub__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return o + (-self)

    def __mul__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Product(self, o)
    
    def __rmul__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Product(o, self)
    
    def __div__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Division(self, o)
    
    def __rdiv__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Division(o, self)
    
    def __pow__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Power(self, o)
    
    def __rpow__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Power(o, self)
    
    def __mod__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Mod(self, o)
    
    def __rmod__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Mod(o, self)
    
    def __neg__(self):
        return -1*self
    
    def __abs__(self):
        return Abs(self)

    def _transpose(self):
        return Transpose(self)
    
    T = property(_transpose)

    #--- Indexing ---

    def __getitem__(self, key):
        return Indexed(self, key)
    
    #--- Differentiation ---
    
    def dx(self, i):
        """Return the partial derivative with respect to spatial variable number i"""
        return PartialDerivative(self, i)
    
    #--- Special functions used for processing expressions ---
    
    def __hash__(self):
        return repr(self).__hash__()
    
    def __eq__(self, other): # alternative to above functions
        return repr(self) == repr(other)
    
    def __contains__(self, item):
        """Return whether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        if isinstance(item, UFLObject):
            if item is self:
                return True
            item = repr(item)
        if repr(self) == item:
            return True
        return any((item in o) for o in self.operands())

#--- Basic terminal objects ---

class Terminal(UFLObject):
    """A terminal node in the expression tree."""

    # Freeze member variables (there are none) for objects of this class
    __slots__ = tuple()
    
    def operands(self):
        return tuple()

class Number(Terminal):
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

# TODO: Should we allow a single Variable to represent a tensor expression?
class Variable(UFLObject):
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

#--- Algebraic operators ---

class Sum(UFLObject):
    __slots__ = ("_operands",)
    
    def __init__(self, *operands):
        ufl_assert(all(operands[0].rank()         == o.rank()         for o in operands), "Rank mismatch in sum.")
        ufl_assert(all(operands[0].free_indices() == o.free_indices() for o in operands), "Can't add expressions with different free indices.")
        self._operands = tuple(operands)
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._operands[0].free_indices()
    
    def rank(self):
        return self._operands[0].rank()
    
    def __str__(self):
        return "(%s)" % " + ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " + ".join(repr(o) for o in self._operands)

class Product(UFLObject):
    __slots__ = ("_operands", "_rank", "_free_indices", "_repeated_indices")
    
    def __init__(self, *operands):
        self._operands = tuple(operands)
       
        # Extract indices
        all_indices = tuple(chain(*(o.free_indices() for o in operands)))
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = extract_indices(all_indices)
        self._free_indices       = free_indices
        self._repeated_indices   = repeated_indices

        # Try to determine rank of this sequence of
        # products with possibly varying ranks of each operand.
        # Products currently defined as valid are:
        # - something multiplied with a scalar
        # - a scalar multiplied with something
        # - matrix-matrix (A*B, M*grad(u))
        # - matrix-vector (A*v)
        current_rank = operands[0].rank()
        for o in operands[1:]:
            if current_rank == 0 or o.rank() == 0:
                # at least one scalar
                current_rank = current_rank + o.rank()
            elif current_rank == 2 and o.rank() == 2:
                # matrix-matrix product
                current_rank = 2
            elif current_rank == 2 and o.rank() == 1:
                # matrix-vector product
                current_rank = 1
            else:
                ufl_error("Invalid combination of tensor ranks in product.")
        
        self._rank = current_rank
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self._rank
    
    def __str__(self):
        return "(%s)" % " * ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " * ".join(repr(o) for o in self._operands)

class Division(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(b), "Division by non-scalar.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._a.free_indices()
    
    def rank(self):
        return self._a.rank()
    
    def __str__(self):
        return "(%s / %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%s / %s)" % (repr(self._a), repr(self._b))

class Power(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar power not defined.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "(%s ** %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%s ** %s)" % (repr(self._a), repr(self._b))
    
class Mod(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar mod not defined.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "(%s %% %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%s %% %s)" % (repr(self._a), repr(self._b))

class Abs(UFLObject):
    __slots__ = ("_a",)
    
    def __init__(self, a):
        self._a = a
    
    def operands(self):
        return (self._a, )
    
    def free_indices(self):
        return self._a.free_indices()
    
    def rank(self):
        return self._a.rank()
    
    def __str__(self):
        return "| %s |" % str(self._a)
    
    def __repr__(self):
        return "Abs(%s)" % repr(self._a)

class Transpose(UFLObject):
    __slots__ = ("_A",)
    
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Transpose is only defined for rank 2 tensors.")
        self._A = A
    
    def operands(self):
        return (self._A,)
    
    def free_indices(self):
        return self._A.free_indices()
    
    def rank(self):
        return 2
    
    def __str__(self):
        return "(%s)^T" % str(self._A)
    
    def __repr__(self):
        return "Transpose(%s)" % repr(self._A)

#--- Indexing ---

class Indexed(UFLObject):
    __slots__ = ("_expression", "_indices", "_fixed_indices", "_free_indices", "_repeated_indices", "_rank")
    
    def __init__(self, expression, indices):
        self._expression = expression
        
        if isinstance(indices, MultiIndex): # if constructed from repr
            self._indices = indices
        else:
            self._indices = MultiIndex(indices, expression.rank())
        
        msg = "Invalid number of indices (%d) for tensor expression of rank %d:\n\t%s\n" % (len(self._indices), expression.rank(), repr(expression))
        ufl_assert(expression.rank() == len(self._indices), msg)
        
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = extract_indices(self._indices._indices)
        # FIXME: We don't need to store all these here, remove the ones we don't use after implementing summation expansion.
        self._fixed_indices      = fixed_indices
        self._free_indices       = free_indices
        self._repeated_indices   = repeated_indices
        self._rank = num_unassigned_indices
    
    def operands(self):
        return tuple(self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self._rank
    
    def __str__(self):
        return "%s[%s]" % (str(self._expression), str(self._indices))
    
    def __repr__(self):
        return "Indexed(%s, %s)" % (repr(self._expression), repr(self._indices))
    
    def __getitem__(self, key):
        ufl_error("Object is already indexed: %s" % repr(self))

#--- Differentiation ---

# FIXME: Add SpatialDerivative and TimeDerivative?

class PartialDerivative(UFLObject):
    "Partial derivative of an expression w.r.t. a spatial direction given by an index."
    
    __slots__ = ("_expression", "_index", "_free_indices") #, "_fixed_indices", "_repeated_indices")
    
    def __init__(self, expression, i):
        self._expression = expression
        self._index = as_index(i)
        
        ufl_assert(not self._index is Axis, "Can't take partial derivative w.r.t. whole axis.")
        
        indices = tuple( [self._index] + expression.free_indices() )
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = extract_indices( indices )
        self._free_indices = free_indices
        
        # We probably don't need these here, remove when sure.
        #self._fixed_indices    = fixed_indices
        #self._repeated_indices = repeated_indices
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self._expression.rank()
    
    def __str__(self):
        return "(d[%s] / dx_%s)" % (str(self._expression), str(self._index))
    
    def __repr__(self):
        return "PartialDerivative(%s, %s)" % (repr(self._expression), repr(self._index))

# FIXME: Anders: Can't we just remove this?

# FIXME: this is just like PartialDiff, should have
#        the exact same behaviour or even be the same class.
class Diff(UFLObject):
    __slots__ = ("f", "x", "_free_indices")

    def __init__(self, f, x):
        self.f = f
        self.x = x
        ufl_assert(is_symbol(x), "Expecting a Symbol in Diff.")
        fi = f.free_indices()
        xi = x.free_indices()
        ufl_assert(len(set(fi) ^ set(xi)) == 0, "Repeated indices in Diff NOT IMPLEMENTED. FIXME!")
        self._free_indices = tuple(fi + xi)
    
    def operands(self):
        return (self.f, self.x)
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self.f.rank()
    
    def __str__(self):
        return "(d[%s] / d[%s])" % (str(self._expression), str(self._index))

    def __repr__(self):
        return "Diff(%s, %s)" % repr(self.f), repr(self.x)

def diff(f, x):
    return Diff(f, x)
