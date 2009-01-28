"""This module defines the IndexSum class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-01-28"

from ufl.expr import Expr
from ufl.indexing import Index, MultiIndex
from ufl.assertions import ufl_assert, assert_expr, assert_instance

#--- Sum over an index ---

class IndexSum(Expr):
    __slots__ = ("_summand", "_index", "_repr")
    
    def __init__(self, summand, index):
        Expr.__init__(self)
        assert_expr(summand)
        if isinstance(index, Index):
            index = MultiIndex((index,))
        assert_instance(index, MultiIndex)
        ufl_assert(len(index) == 1, "Expecting a single Index only.")
        self._summand = summand
        self._index = index
        self._repr = "IndexSum(%r, %r)" % (summand, index)
    
    def operands(self):
        return (self._summand, self._index)
    
    def indices(self):
        j = self._index[0]
        return tuple(i for i in self._summand.free_indices() if not i == j)
    
    def index_dimensions(self):
        return self._operands[0].index_dimensions()
    
    def shape(self):
        return self._summand.shape()
    
    def evaluate(self, x, mapping, component, index_values):
        return sum(o.evaluate(x, mapping, component, index_values) for o in self.operands())
    
    def __str__(self):
        return "sum_{%s}< %s >" % (str(self._index), str(self._summand))
    
    def __repr__(self):
        return self._repr

