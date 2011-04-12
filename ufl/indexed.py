"""This module defines the Indexed class."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2009-01-28 -- 2009-04-19"

from collections import defaultdict
from ufl.log import error
from ufl.expr import Expr, WrapperType
from ufl.indexing import IndexBase, Index, as_multi_index
from ufl.indexutils import unique_indices
from ufl.precedence import parstr

#--- Indexed expression ---

class Indexed(WrapperType):
    __slots__ = ("_expression", "_indices",
                 "_free_indices", "_index_dimensions", "_repr")
    def __init__(self, expression, indices):
        WrapperType.__init__(self)
        if not isinstance(expression, Expr):
            error("Expecting Expr instance, not %s." % repr(expression))
        self._expression = expression
        shape = expression.shape()
        self._indices = as_multi_index(indices, shape)
        
        if expression.rank() != len(self._indices):
            error("Invalid number of indices (%d) for tensor "\
                "expression of rank %d:\n\t%r\n"\
                % (len(self._indices), expression.rank(), expression))
        
        idims = dict((i, s) for (i, s) in zip(self._indices._indices, shape) if isinstance(i, Index))
        idims.update(expression.index_dimensions())
        fi = unique_indices(expression.free_indices() + self._indices._indices)
        
        self._free_indices = fi
        self._index_dimensions = idims

        self._repr = "Indexed(%r, %r)" % (self._expression, self._indices)
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return ()
    
    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        A, ii = self.operands()
        component = ii.evaluate(x, mapping, None, index_values)
        if derivatives:
            return A.evaluate(x, mapping, component, index_values, derivatives)
        return A.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "%s[%s]" % (parstr(self._expression, self), self._indices)
    
    def __repr__(self):
        return self._repr
    
    def __getitem__(self, key):
        error("Attempting to index with %r, but object is already indexed: %r" % (key, self))

