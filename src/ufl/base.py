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

def is_scalar_valued(o):
    return o.rank == 0

def is_true_scalar(o):
    return o.rank == 0  and  len(o.free_indices) == 0

def free_indices(o):
    return o.free_indices


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
        ufl_assert(A.rank() == 2, "Transpose is only defined for rank 2 tensors.")
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
                ufl_assert(v <= 2, "Index %s repeated %d times." % (str(k), v))
        
        self.repeated_indices = tuple(rep)
        self.free_indices     = tuple(free)
    
    def operands(self):
        return self._operands
    
    def __repr__(self):
        return "(%s)" % " * ".join(repr(o) for o in self._operands)


class Sum(UFLObject):
    def __init__(self, *operands):
        self._operands = tuple(operands)
        
        r = operands[0].rank()
        self.rank = r
        ufl_assert(all(r == o.rank() for o in operands), "Rank mismatch in sum.")
        
        # create new (relabel) indices unless all indices of all operands are equal
        if all(o.free_indices == operands[0].free_indices for o in operands):
            self.free_indices = operands[0].free_indices
        else:
            self.free_indices = tuple(Index() for i in range(r))
    
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
    
    def __repr__(self):
        return "Index(%s, %d)" % (repr(self.name), self.count)


class MultiIndex(UFLObject):
    def __init__(self, indices): # FIXME: make operands and constructor consistent here
        if isinstance(indices, tuple):
            self.indices = indices
        elif isinstance(indices, (Index,Integer,int)): # TODO: Might have to wrap int in Integer class, for consistent expression tree traversal.
            self.indices = (indices,)
        else:
            raise UFLException("Expecting Index, or Integer objects.")
    
    def __repr__(self):
        return "MultiIndex(%s)" % repr(self.indices)

    def __len__(self):
        return len(self.indices)

    def operands(self):
        return self.indices

class Indexed(UFLObject):
    def __init__(self, expression, indices):
        self.expression = expression
        if isinstance(indices, MultiIndex):
            self.indices = indices
        else:
            self.indices = MultiIndex(indices)
        ufl_assert(expression.rank() == len(self.indices), "Invalid number of indices (%d) for tensor of rank %d." % (len(self.indices), expression.rank()))
        #self.free_indices = tuple(i for i in self.indices if isinstance(i, Index)) # FIXME
        self.free_indices = tuple() # FIXME
        self.rank = expression.rank
    
    def __repr__(self):
        return "Indexed(%s, %s)" % (repr(self.expression), repr(self.indices))
    
    def __getitem__(self, key):
        ufl_assert(False, "Object is already indexed: %s" % repr(self))
    
    def operands(self):
        return tuple(self.expression, self.indices)



### How to handle tensor, subcomponents, indexing, Einstein summation? TODO: Need experiences from FFC!


if __name__ == "__main__":
    print "No tests here."

