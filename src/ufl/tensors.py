
from base import *

# TODO: support list of expressions
class Tensor(UFLObject):
    def __init__(self, expressions, indices):
        if isinstance(expressions, list):
            raise NotImplemented("")
        else:
            ufl_assert(expressions.rank() == 0, "Need scalar valued expressions.")
            ufl_assert(all(i in indices for i in expressions.free_indices()), "Index set mismatch.")
        self._expressions = expressions
        self._indices    = indices
        self._free_indices = tuple(set(expressions.free_indices()) - set(indices))
    
    def rank(self):
        return len(self._indices)
    
    def free_indices(self):
        return self._free_indices


# TODO: support list of expressions
class Matrix(Tensor):
    def __init__(self, expressions, indices = None):
        if isinstance(expressions, list):
            raise NotImplemented("")
        else:
            ufl_assert(len(indices) == 2, "Need two indices for a matrix.")
        Tensor.__init__(self, expressions, indices)


# TODO: support list of expressions
class Vector(Tensor):
    def __init__(self, expressions, index):
        if isinstance(expressions, list):
            raise NotImplemented("")
        else:
            ufl_assert(isinstance(index, Index), "Need one Index for a vector.")
        Tensor.__init__(self, expressions, (index,))


