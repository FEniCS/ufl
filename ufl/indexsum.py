"""This module defines the IndexSum class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-02-04"

from ufl.assertions import ufl_assert, assert_expr, assert_instance
from ufl.indexing import Index, MultiIndex, as_multi_index
from ufl.expr import AlgebraOperator

#--- Sum over an index ---

class IndexSum(AlgebraOperator):
    __slots__ = ("_summand", "_index", "_repr", "_free_indices", "_index_dimensions")
    
    def __init__(self, summand, index):
        AlgebraOperator.__init__(self)
        assert_expr(summand)
        index = as_multi_index(index)
        ufl_assert(len(index) == 1, "Expecting a single Index only.")
        self._summand = summand
        self._index = index
        self._repr = "IndexSum(%r, %r)" % (summand, index)
        
        j = self._index[0]
        self._free_indices = tuple(i for i in self._summand.free_indices() if not i == j)
        self._index_dimensions = dict(self._summand.index_dimensions())
        del self._index_dimensions[j]
    
    def operands(self):
        return (self._summand, self._index)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._summand.shape()
    
    def evaluate(self, x, mapping, component, index_values):
        d = self._summand.index_dimensions()[self._index]
        tmp = 0
        for i in range(d):
            index_values.push(self._index, i)
            tmp += self._summand.evaluate(x, mapping, component, index_values)
            index_values.pop()
        return tmp

    def __str__(self):
        return "sum_{%s}< %s >" % (str(self._index), str(self._summand))
    
    def __repr__(self):
        return self._repr

