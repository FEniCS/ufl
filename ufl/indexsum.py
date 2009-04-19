"""This module defines the IndexSum class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-04-19"

from ufl.log import error
from ufl.expr import Expr, AlgebraOperator
from ufl.indexing import Index, MultiIndex, as_multi_index
from ufl.precedence import parstr

#--- Sum over an index ---

class IndexSum(AlgebraOperator):
    __slots__ = ("_summand", "_index", "_dimension", "_repr", "_free_indices", "_index_dimensions")
    
    def __new__(cls, summand, index):
        from ufl.constantvalue import Zero
        if isinstance(summand, Zero):
            sh = summand.shape()
            j, = index
            fi = tuple(i for i in summand.free_indices() if not i == j)
            idims = dict(summand.index_dimensions())
            del idims[j]
            return Zero(sh, fi, idims)
        return AlgebraOperator.__new__(cls)

    def __init__(self, summand, index):
        AlgebraOperator.__init__(self)
        if not isinstance(summand, Expr):
            error("Expecting Expr instance, not %s." % repr(summand))
        
        if isinstance(index, Index):
            j = index
        elif isinstance(index, MultiIndex):
            if len(index) != 1:
                error("Expecting a single Index only.")
            j, = index

        self._summand = summand
        self._index_dimensions = dict(summand.index_dimensions())
        self._free_indices = tuple(i for i in summand.free_indices() if not i == j)

        d = self._index_dimensions[j]
        self._index = as_multi_index(index, (d,))
        self._dimension = d
        del self._index_dimensions[j]

        self._repr = "IndexSum(%r, %r)" % (summand, index)
    
    def dimension(self):
        return self._dimension
    
    def operands(self):
        return (self._summand, self._index)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._summand.shape()
    
    def evaluate(self, x, mapping, component, index_values):
        i, = self._index
        tmp = 0
        for k in range(self._dimension):
            index_values.push(i, k)
            tmp += self._summand.evaluate(x, mapping, component, index_values)
            index_values.pop()
        return tmp

    def __str__(self):
        return "sum_{%s} %s " % (str(self._index), parstr(self._summand, self))
    
    def __repr__(self):
        return self._repr

