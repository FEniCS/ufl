#!/usr/bin/env python

"""
This module contains the UFLObject base class and all expression
types involved with built-in operators on any ufl object.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 11th 2008"

import operator
from itertools import chain
from collections import defaultdict

### Utility functions:

class UFLException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

def ufl_assert(condition, message):
    if not condition:
        raise UFLException(message)

def is_python_scalar(o):
    return isinstance(o, (int, float))

def product(l):
    return reduce(operator.__mul__, l)


# ... Helper functions for tensor properties:

def rank(o):
    return o.rank

def free_indices(o):
    return o.free_indices

def is_scalar_valued(o):
    return o.rank == 0

def is_true_scalar(o):
    return o.rank == 0  and  len(o.free_indices) == 0


### UFLObject base class:

class UFLObjectBase(object):
    """Interface or ufl objects, all classes should implement these."""
    def __init__(self):
        # all classes should define these variables
        self.free_indices = None
        self.rank = None
    
    # ... Access to subtree nodes for expression traversal:
    
    def operands(self):
        """Returns a sequence with all subtree nodes in expression tree.
           All UFLObject subclasses are required to implement operands ."""
        raise NotImplementedError(self.__class__.operands)
    
    # ... Representation strings are required:
    
    def __repr__(self):
        """It is required to implement repr for all UFLObject subclasses."""
        raise NotImplementedError(self.__class__.__repr__)


class UFLObject(UFLObjectBase):
    """An UFLObject is equipped with all relevant operators."""
    def __init__(self):
        pass
    
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
    
    def transpose(self):
        return Transpose(self)
    
    T = property(transpose)
    
    # ... Indexing a tensor, or relabeling the indices of a tensor
    
    def __getitem__(self, key):
        return Indexed(self, key)
    
    # ... Strings:
    
    def __str__(self):
        return repr(self)
    
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
    def __init__(self):
        pass
    
    def operands(self):
        return tuple()


class Integer(Terminal):
    def __init__(self, value):
        self.value = value
        self.free_indices = tuple()
        self.rank = 0
    
    def __repr__(self):
        return "Integer(%s)" % repr(self.value)


class Real(Terminal): # TODO: Do we need this? Numeric tensors?
    def __init__(self, value):
        self.value = value
        self.free_indices = tuple()
        self.rank = 0
    
    def __repr__(self):
        return "Real(%s)" % repr(self.value)


class Number(Terminal):
    def __init__(self, value):
        self.value = value
        self.free_indices = tuple()
        self.rank = 0
    
    def __repr__(self):
        return "Number(%s)" % repr(self.value)


class Identity(Terminal):
    def __init__(self):
        pass
    
    def __repr__(self):
        return "Identity()"


class Symbol(Terminal): # TODO: Needed for diff? Tensors of symbols? Parametric symbols?
    def __init__(self, name):
        self.name = name
        self.free_indices = tuple()
        self.rank = 0
    
    def __repr__(self):
        return "Symbol(%s)" % repr(self.name)


#class Variable(UFLObject): # TODO: what is this really?
#    def __init__(self, name, expression):
#        self.name = name # TODO: must wrap in UFLString or UFLName or something
#        self.expression = expression
#    
#    def operands(self):
#        return (self.name, self.expression)
#    
#    def __repr__(self):
#        return "Variable(%s, %s)" % (repr(self.name), repr(self.expression))



### Algebraic operators

class Transpose(UFLObject):
    def __init__(self, A):
        ufl_assert(rank(A) == 2, "Transpose is only defined for rank 2 tensors.")
        self.A = A
        self.free_indices = A.free_indices
        self.rank = 2
    
    def operands(self):
        return (self.A,)
    
    def __repr__(self):
        return "Transpose(%s)" % repr(self.A)


class Product(UFLObject):
    def __init__(self, *operands):
        self._operands = tuple(operands)
        
        rep  = []
        free = []
        
        count = defaultdict(int)
        for i in chain(o.free_indices for o in operands):
            count[i] += 1
        
        for k, v in count.iteritems():
            if v == 1:
                free.append(k)
            elif v == 2:
                rep.append(k)
            else:
                ufl_assert(v <= 2, "Undefined behaviour: Index %s is repeated %d times." % (str(k), v))
        
        # remember repeated indices for later summation, not sure where to use this yet
        self.repeated_indices = tuple(rep)
        self.free_indices     = tuple(free)
        
        # Try to determine rank of this sequence of
        # products with possibly varying ranks of each operand.
        # Products currently defined as valid are:
        # - something multiplied with a scalar
        # - a scalar multiplied with something
        # - matrix-matrix (A*B, M*grad(u))
        # - matrix-vector (A*v)
        current_rank = operands[0].rank
        for o in operands[1:]:
            if current_rank == 0 or o.rank == 0:
                # at least one scalar
                current_rank = current_rank + o.rank
            elif current_rank == 2 and o.rank == 2:
                # matrix-matrix product
                current_rank = 2
            elif current_rank == 2 and o.rank == 1:
                # matrix-vector product
                current_rank = 1
            else:
                ufl_assert(False, "Invalid combination of tensor ranks in product.")
        
        self.rank = current_rank
    
    def operands(self):
        return self._operands
    
    def __repr__(self):
        return "(%s)" % " * ".join(repr(o) for o in self._operands)


class Sum(UFLObject):
    def __init__(self, *operands):
        r = operands[0].rank
        ufl_assert(all(r == o.rank for o in operands), "Rank mismatch in sum.")
        ufl_assert(all(o.free_indices == operands[0].free_indices for o in operands), "Can't add expressions with different free indices.")
        
        self._operands = tuple(operands)
        self.rank = r
        self.free_indices = operands[0].free_indices
    
    def operands(self):
        return self._operands
    
    def __repr__(self):
        return "(%s)" % " + ".join(repr(o) for o in self._operands)


class Division(UFLObject):
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(b), "Division by non-scalar.")
        self.a = a
        self.b = b
        self.free_indices = a.free_indices
        self.rank = a.rank
    
    def operands(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s / %s)" % (repr(self.a), repr(self.b))


class Power(UFLObject):
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar power not defined.")
        self.a = a
        self.b = b
        self.free_indices = tuple()
        self.rank = 0
    
    def operands(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s ** %s)" % (repr(self.a), repr(self.b))
    

class Mod(UFLObject):
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar mod not defined.")
        self.a = a
        self.b = b
        self.free_indices = tuple()
        self.rank = 0
    
    def operands(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "(%s %% %s)" % (repr(self.a), repr(self.b))


class Abs(UFLObject):
    def __init__(self, a):
        self.a = a
        self.free_indices = a.free_indices
        self.rank = a.rank
    
    def operands(self):
        return (self.a, )
    
    def __repr__(self):
        return "Abs(%s)" % repr(self.a)
    


### Indexing

class Index(Terminal):
    count = 0
    def __init__(self, name = None, count = None):
        self.name = name
        if count is None:
            self.count = Index.count
            Index.count += 1
        else:
            self.count = count # TODO: modify Index.count, similarly in Function etc.
        # these make no sense here:
        self.rank = None
        self.free_indices = None
    
    def __repr__(self):
        return "Index(%s, %d)" % (repr(self.name), self.count)


class FixedIndex(Terminal):
    def __init__(self, value):
        ufl_assert(isinstance(value, int), "Expecting integer value for fixed index.")
        self.value = value
        # these make no sense here:
        self.rank = None
        self.free_indices = None
    
    def __repr__(self):
        return "FixedIndex(%d)" % self.value


class MultiIndex(UFLObject):
    def __init__(self, indices):
        # use a tuple consistently
        if not isinstance(indices, tuple):
            indices = (indices,)
        # use Index or FixedIndex consistently
        ind = []
        for i in indices:
            if isinstance(i, Index):
                ind.append(i)
            elif isinstance(i, int):
                ind.append(FixedIndex(i))
            else:
                ufl_assert(False, "Unfamiliar index object %s" % repr(i))
        self.indices = tuple(ind)
        # these make no sense here:
        self.rank = None
        self.free_indices = None
    
    def operands(self):
        return self.indices
    
    def __repr__(self):
        return "MultiIndex(%s)" % repr(self.indices)

    def __len__(self):
        return len(self.indices)


class Indexed(UFLObject):
    def __init__(self, expression, indices):
        self.expression = expression
        
        if isinstance(indices, MultiIndex):
            self.indices = indices
        else:
            self.indices = MultiIndex(indices)
        
        msg = "Invalid number of indices (%d) for tensor expression of rank %d:\n\t%s\n" % (len(self.indices), expression.rank, repr(expression))
        ufl_assert(expression.rank == len(self.indices), msg)
        
        self.rank = expression.rank - len(self.indices)
        
        self.free_indices = tuple(i for i in self.indices.indices if isinstance(i, Index))
    
    def operands(self):
        return tuple(self.expression, self.indices)
    
    def __repr__(self):
        return "Indexed(%s, %s)" % (repr(self.expression), repr(self.indices))
    
    def __getitem__(self, key):
        ufl_assert(False, "Object is already indexed: %s" % repr(self))



### How to handle tensor, subcomponents, indexing, Einstein summation? TODO: Need experiences from FFC!


if __name__ == "__main__":
    print "No tests here."

