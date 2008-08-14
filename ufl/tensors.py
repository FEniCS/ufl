"""Classes used to group scalar expressions into expressions with rank > 0."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-31 -- 2008-08-14"


from .output import ufl_assert
from .base import UFLObject, Terminal
from .indexing import Index


class ListVector(UFLObject):
    __slots__ = ("_expressions", "_free_indices")
    
    def __init__(self, expressions):
        ufl_assert(isinstance(expressions, list), "Expecting list of expressions.")
        ufl_assert(all(e.rank() == 0 for e in expressions), "Expecting scalar valued expressions.")
        
        eset = set(expressions[0].free_indices())
        self._free_indices = tuple(eset)
        self._expressions  = expressions
        
        ufl_assert(all(len(eset ^ set(e.free_indices())) == 0 for e in expressions), "Can't handle list of expressions with different free indices.")
        #ufl_assert(len(expressions.free_indices()) == 0, "Can't handle list of expressions with free indices.")
    
    def operands(self):
        return (self._expressions,)
    
    def rank(self):
        return 1
    
    def free_indices(self):
        return self._free_indices
    
    def __str__(self):
        return "<%s>" % ", ".join(str(e) for e in self._expressions)
    
    def __repr__(self):
        return "ListVector(%s)" % repr(self._expressions)


class ListMatrix(UFLObject):
    __slots__ = ("_expressions", "_free_indices")
    
    def __init__(self, expressions):
        ufl_assert(isinstance(expressions, list),                 "Expecting list.")
        ufl_assert(all(isinstance(e, list) for e in expressions), "Expecting list of lists of expressions.")
        
        r = len(expressions)
        c = len(expressions[0])
        
        ufl_assert(all(len(row) == c for row in expressions),              "Inconsistent row size.")
        ufl_assert(all(e.rank() == 0 for row in expressions for e in row), "Expecting scalar valued expressions.")
        
        eset = set(expressions[0][0].free_indices())
        self._free_indices = tuple(eset)
        self._expressions  = expressions
        
        ufl_assert(all(len(eset ^ set(e.free_indices())) == 0 for row in expressions for e in row), "Can't handle list of expressions with different free indices.")
        #ufl_assert(len(expressions.free_indices()) == 0,                  "Can't handle list of expressions with free indices.")
    
    def operands(self):
        return (self._expressions,)
    
    def rank(self):
        return 2
    
    def free_indices(self):
        return self._free_indices
    
    def __str__(self):
        rowstrings = []
        for row in self._expressions:
            rowstrings.append( ("[%s]" % ", ".join(str(e) for e in row)) ) 
        return "[ %s ]" % ", ".join(rowstrings)
    
    def __repr__(self):
        return "ListMatrix(%s)" % repr(self._expressions)


class Tensor(UFLObject):
    __slots__ = ("_expressions", "_indices", "_free_indices")
    
    def __init__(self, expressions, indices):
        ufl_assert(isinstance(expressions, UFLObject),          "Expecting ufl expression.")
        ufl_assert(expressions.rank() == 0,                     "Expecting scalar valued expressions.")
        ufl_assert(all(isinstance(i, Index) for i in indices),  "Expecting Index instances in indices list.")
        
        eset = set(expressions.free_indices())
        iset = set(indices)
        ufl_assert(len(iset - eset) == 0,  "Index mismatch.")
        
        self._expressions  = expressions
        self._indices      = indices
        self._free_indices = tuple(eset - iset)
    
    def operands(self):
        return (self._expressions, self._indices)
    
    def rank(self):
        return len(self._indices)
    
    def free_indices(self):
        return self._free_indices

    def __str__(self):
        return "[Rank %d tensor A, such that A_{%s} = %s]" % (self.rank(), self._indices, self._expressions)
    
    def __repr__(self):
        return "Tensor(%s, %s)" % (repr(self._expressions), repr(self._indices))


def Vector(expressions, index = None):
    if index is None:
        return ListVector(expressions)
    return Tensor(expressions, (index,))


def Matrix(expressions, indices = None):
    if indices is None:
        return ListMatrix(expressions)
    return Tensor(expressions, indices)

