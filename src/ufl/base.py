#!/usr/bin/env python

"""
This module contains the UFLObject base class and all expression
types involved with built-in operators on any ufl object.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-14-03"

import operator
from itertools import chain
from collections import defaultdict
from output import *


# FIXME: I've messed up a bit on the name conventions, sometimes using "self._foo" and sometimes "self.foo".
#        This is not consistent and should be fixed.
#        The way to go: "self._foo" for everything the external user shouldn't use, which is most variables.


# ... Utility functions:

def is_python_scalar(o):
    return isinstance(o, (int, float))

def product(l):
    return reduce(operator.__mul__, l)


# ... Helper functions for tensor properties:

def is_scalar_valued(o):
    """Checks if an expression is scalar valued, possibly still with free indices. Returns True/False."""
    ufl_assert(isinstance(o, UFLObject), "Assuming an UFLObject.")
    return o.rank() == 0

def is_true_scalar(o):
    """Checks if an expression represents a single scalar value, with no free indices. Returns True/False."""
    return is_scalar_valued(o) and len(o.free_indices()) == 0


class UFLObject(object):
    """Base class of all UFL objects"""
    
    # Freeze member variables (there are none) of objects of this class.
    __slots__ = tuple()
    
    # ... "Abstract" functions: Functions that subexpressions should implement.
    
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
    
    # All UFL objects should implement __str__, as default we use repr
    def __str__(self):
        """Return pretty print string representation of objects"""
        return repr(self)
    
    # ... Algebraic operators:
    
    def __mul__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Product(self, o)
    
    def __rmul__(self, o):
        if is_python_scalar(o): o = Number(o)
        if not isinstance(o, UFLObject): return NotImplemented
        return Product(o, self)
    
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
    
    # ... Partial derivatives
    
    def dx(self, i):
        """Returns the partial derivative of this expression with respect to spatial variable number i."""
        return PartialDerivative(self, i)
    
    # ... Indexing a tensor, or relabeling the indices of a tensor
    
    def __getitem__(self, key):
        return Indexed(self, key)
    
    # ... Support for inserting an UFLObject in dicts and sets:
    
    def __hash__(self):
        return repr(self).__hash__()
    
    def __eq__(self, other): # alternative to above functions
        return repr(self) == repr(other)
    
    # ... Searching for an UFLObject the subexpression tree:
    
    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        if isinstance(item, UFLObject):
            if item is self:
                return True
            item = repr(item)
        if repr(self) == item:
            return True
        return any((item in o) for o in self.operands())



### Basic terminal objects

class Terminal(UFLObject):
    """A terminal node in the expression tree."""
    
    # Freeze member variables (there are none) of objects of this class.
    __slots__ = tuple()
    
    def operands(self):
        return tuple()


class Number(Terminal):
    __slots__ = ("value",)
    
    def __init__(self, value):
        self.value = value
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return "Number(%s)" % repr(self.value)


class Identity(Terminal):
    __slots__ = tuple()
    
    def __str__(self):
        return "I"
    
    def __repr__(self):
        return "Identity()"


class Symbol(Terminal): # TODO: What about tensors of symbols?
    __slots__ = tuple("name")
    
    def __init__(self, name):
        self.name = str(name)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return "Symbol(%s)" % repr(self.name)


class Variable(UFLObject): # TODO: Use this for diff. What about tensors of variables?
    __slots__ = ("symbol", "expression")
    
    def __init__(self, symbol, expression):
        self.symbol     = symbol if isinstance(symbol, Symbol) else Symbol(symbol)
        self.expression = expression
    
    def operands(self):
        return (self.symbol, self.expression)
    
    def __repr__(self):
        return "Variable(%s, %s)" % (repr(self.symbol), repr(self.expression))



### Algebraic operators

class Transpose(UFLObject):
    __slots__ = ("A",)
    
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Transpose is only defined for rank 2 tensors.")
        self.A = A
    
    def operands(self):
        return (self.A,)
    
    def free_indices(self):
        return A.free_indices()
    
    def rank(self):
        return 2
    
    def __str__(self):
        return "(%s)^T" % str(self.A)
    
    def __repr__(self):
        return "Transpose(%s)" % repr(self.A)


class Product(UFLObject):
    __slots__ = ("_operands", "_rank", "_free_indices", "_repeated_indices")
    
    def __init__(self, *operands):
        self._operands = tuple(operands)
       
        # analyzing free indices
        all_indices = tuple(chain(*(o.free_indices() for o in operands)))
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = analyze_indices(all_indices)
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


class Division(UFLObject):
    __slots__ = ("a", "b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(b), "Division by non-scalar.")
        self.a = a
        self.b = b
    
    def operands(self):
        return (self.a, self.b)
    
    def free_indices(self):
        return self.a.free_indices()
    
    def rank(self):
        return self.a.rank()
    
    def __str__(self):
        return "(%s / %s)" % (str(self.a), str(self.b))
    
    def __repr__(self):
        return "(%s / %s)" % (repr(self.a), repr(self.b))


class Power(UFLObject):
    __slots__ = ("a", "b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar power not defined.")
        self.a = a
        self.b = b
    
    def operands(self):
        return (self.a, self.b)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "(%s ** %s)" % (str(self.a), str(self.b))
    
    def __repr__(self):
        return "(%s ** %s)" % (repr(self.a), repr(self.b))
    

class Mod(UFLObject):
    __slots__ = ("a", "b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar mod not defined.")
        self.a = a
        self.b = b
    
    def operands(self):
        return (self.a, self.b)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "(%s %% %s)" % (str(self.a), str(self.b))
    
    def __repr__(self):
        return "(%s %% %s)" % (repr(self.a), repr(self.b))


class Abs(UFLObject):
    __slots__ = ("a",)
    
    def __init__(self, a):
        self.a = a
    
    def operands(self):
        return (self.a, )
    
    def free_indices(self):
        return self.a.free_indices()
    
    def rank(self):
        return self.a.rank()
    
    def __str__(self):
        return "| %s |" % str(self.a)
    
    def __repr__(self):
        return "Abs(%s)" % repr(self.a)
    


### Indexing

class Index(Terminal):
    __slots__ = ("name", "count")
    
    _globalcount = 0
    def __init__(self, name = None, count = None):
        self.name = name
        if count is None:
            self.count = Index._globalcount
            Index._globalcount += 1
        else:
            self.count = count
            if count >= Index._globalcount:
                Index._globalcount = count + 1
    
    def free_indices(self):
        ufl_error("Why would you want to get the free indices of an Index? Please explain at ufl-dev@fenics.org...")
    
    def rank(self):
        ufl_error("Why would you want to get the rank of an Index? Please explain at ufl-dev@fenics.org...")
    
    def __str__(self):
        return "i_%d" % self.count # TODO: use name? Maybe just remove name, adds possible confusion of what ID's an Index (which is the count alone).
    
    def __repr__(self):
        return "Index(%s, %d)" % (repr(self.name), self.count)


class FixedIndex(Terminal):
    __slots__ = ("value",)
    
    def __init__(self, value):
        ufl_assert(isinstance(value, int), "Expecting integer value for fixed index.")
        self.value = value
    
    def free_indices(self):
        ufl_error("Why would you want to get the free indices of an Index? Please explain at ufl-dev@fenics.org...")
    
    def rank(self):
        ufl_error("Why would you want to get the rank of an Index? Please explain at ufl-dev@fenics.org...")
    
    def __repr__(self):
        return "FixedIndex(%d)" % self.value


def analyze_indices(indices):
    ufl_assert(isinstance(indices, tuple), "Assuming index tuple.")
    
    count = defaultdict(int)
    for i in indices:
        count[i] += 1
    
    unique_indices = count.keys()

    fixed_indices      = []
    free_indices       = []
    repeated_indices   = []
    num_unassigned_indices = 0
    
    for i in unique_indices:
        if isinstance(i, int):
            fixed_indices.append(FixedIndex(i))
        elif isinstance(i, FixedIndex):
            fixed_indices.append(i)
        elif isinstance(i, Index):
            c = count[i]
            if c == 1:
                free_indices.append(i)
            elif c == 2:
                repeated_indices.append(i)
            else:
                ufl_error("Invalid index repetition count %d" % c)
        elif isinstance(i, slice):
            if i.start is None and i.stop is None and i.step is None:
                num_unassigned_indices += count[i]
            else:
                ufl_error("Can't handle specific slice, only general ':'.")
        elif i is Ellipsis: # '...' as in A[i, :, 0, ..., 1]
            ufl_error("Can't handle ellipsis.")
        else:
            ufl_error("Invalid index type %s" % i.__class__)
    
    fixed_indices      = tuple(fixed_indices)
    free_indices       = tuple(free_indices)
    repeated_indices   = tuple(repeated_indices)
    
    ufl_assert(len(fixed_indices) + len(free_indices) + 2*len(repeated_indices) + num_unassigned_indices == len(indices), "Logic breach in analyze_indices.")
    
    return (fixed_indices, free_indices, repeated_indices, num_unassigned_indices)


def as_index(i): # TODO: handle ":" as well!
    """Takes something the user might input as an index, and returns an actual UFL index object."""
    if isinstance(i, (Index, FixedIndex)):
        return i
    elif isinstance(i, int):
        return FixedIndex(i)
    else:
        ufl_error("Can convert this object to index: %s" % repr(i))


class MultiIndex(UFLObject):
    __slots__ = ("indices",)
    
    def __init__(self, indices):
        # make a consistent tuple of Index and FixedIndex objects
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.indices = tuple(as_index(i) for i in indices)
    
    def operands(self):
        return self.indices
    
    def free_indices(self):
        ufl_error("Why would you want to get the free indices of a MultiIndex? Please explain at ufl-dev@fenics.org...")
    
    def rank(self):
        ufl_error("Why would you want to get the rank of a MultiIndex? Please explain at ufl-dev@fenics.org...")
    
    def __str__(self):
        return ", ".join(str(i) for i in self.indices)
    
    def __repr__(self):
        return "MultiIndex(%s)" % repr(self.indices)

    def __len__(self):
        return len(self.indices)


class Indexed(UFLObject):
    __slots__ = ("_expression", "_indices", "_fixed_indices", "_free_indices", "_repeated_indices", "_rank")
    
    def __init__(self, expression, indices):
        self._expression = expression
        
        if isinstance(indices, MultiIndex): # if constructed from repr?
            self._indices = indices
        else:
            self._indices = MultiIndex(indices)
        
        msg = "Invalid number of indices (%d) for tensor expression of rank %d:\n\t%s\n" % (len(self._indices), expression.rank(), repr(expression))
        ufl_assert(expression.rank() == len(self._indices), msg)
        
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = analyze_indices(self._indices.indices)
        # FIXME: We don't need to store all these here, remove the ones we don't use.
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


### Derivatives

class PartialDerivative(UFLObject):
    "Partial derivative of an expression w.r.t. a spatial direction given by an index."
    
    __slots__ = ("expression", "index", "_fixed_indices", "_free_indices", "_repeated_indices")
    
    def __init__(self, expression, i):
        self.expression = expression
        self.index = as_index(i)
        
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = analyze_indices( tuple( [self.index] + expression.free_indices() ) )
        # FIXME: We don't need all these here, remove the ones we don't use.
        #self._fixed_indices      = fixed_indices
        self._free_indices       = free_indices
        #self._repeated_indices   = repeated_indices
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self.expression.rank()
    
    def __str__(self):
        return "(d[%s] / dx_%s)" % (str(self.expression), str(self.index))
    
    def __repr__(self):
        return "PartialDerivative(%s, %s)" % (repr(self.expression), repr(self.index))


