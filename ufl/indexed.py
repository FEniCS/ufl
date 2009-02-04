"""This module defines the Indexed class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-02-03"

from collections import defaultdict
from ufl.log import error
from ufl.expr import WrapperType
from ufl.indexing import IndexBase, Index, FixedIndex, MultiIndex, as_multi_index
from ufl.indexutils import unique_indices
from ufl.assertions import ufl_assert, assert_expr, assert_instance

#--- Indexed expression ---

class Indexed(WrapperType):
    __slots__ = ("_expression", "_indices",
                 "_free_indices", "_index_dimensions")
    def __init__(self, expression, indices):
        WrapperType.__init__(self)
        assert_expr(expression)
        self._expression = expression
        self._indices = as_multi_index(indices)
        
        n = len(self._indices)
        r = expression.rank()
        msg = "Invalid number of indices (%d) for tensor "\
              "expression of rank %d:\n\t%r\n" % (n, r, expression)
        ufl_assert(r == n, msg)
        
        shape = expression.shape()
        idims = dict((i, s) for (i, s) in zip(self._indices._indices, shape))
        idims.update(expression.index_dimensions())
        fi = unique_indices(expression.free_indices() + self._indices._indices)
        
        self._free_indices = fi
        self._index_dimensions = idims
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return ()
    
    def evaluate(self, x, mapping, component, index_values):
        A, ii = self.operands()
        
        component = ii.evaluate(x, mapping, None, index_values)
        return A.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "%s[%s]" % (self._expression, self._indices)
    
    def __repr__(self):
        return "Indexed(%r, %r)" % (self._expression, self._indices)
    
    def __getitem__(self, key):
        error("Attempting to index with %r, but object is already indexed: %r" % (key, self))

