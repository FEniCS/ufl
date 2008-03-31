

from base import *


class ListVector(UFLObject):
    def __init__(self, expressions):
        ufl_assert(isinstance(expressions, list), "Expecting list of expressions.")
        ufl_assert(all(e.rank() == 0 for e in expressions), "Expecting scalar valued expressions.")
        
        eset = set(expressions[0].free_indices())
        self._free_indices = tuple(eset)
        self._expressions  = expressions
        
        ufl_assert(all(len(eset ^ set(e.free_indices())) == 0 for e in expressions), "Can't handle list of expressions with different free indices.")
        #ufl_assert(len(expressions.free_indices()) == 0, "Can't handle list of expressions with free indices.")
    
    def rank(self):
        return 1
    
    def free_indices(self):
        return self._free_indices


class ListMatrix(UFLObject):
    def __init__(self, expressions):
        ufl_assert(isinstance(expressions, list),                 "Expecting list.")
        ufl_assert(all(isinstance(e, list) for e in expressions), "Expecting list of lists of expressions.")
        
        r = len(expressions)
        c = len(expressions[0])
        
        ufl_assert(all(len(row) == c for row in expressions),              "Inconsistent row size.")
        ufl_assert(all(e.rank() == 0 for e in row for row in expressions), "Expecting scalar valued expressions.")
        
        eset = set(expressions[0].free_indices())
        self._free_indices = tuple(eset)
        self._expressions  = expressions
        
        ufl_assert(all(eset ^ set(e.free_indices()) for e in expressions), "Can't handle list of expressions with different free indices.")
        #ufl_assert(len(expressions.free_indices()) == 0,                  "Can't handle list of expressions with free indices.")
    
    def rank(self):
        return 2
    
    def free_indices(self):
        return self._free_indices


class IndexTensor(UFLObject):
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
    
    def rank(self):
        return len(self._indices)
    
    def free_indices(self):
        return self._free_indices


def Vector(expressions, index = None):
    if index is None:
        return ListVector(expressions)
    return IndexTensor(expressions, (index,))


def Matrix(expressions, indices = None):
    if indices is None:
        return ListMatrix(expressions)
    return IndexTensor(expressions, indices)



