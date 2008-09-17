"""Classes used to group scalar expressions into expressions with rank > 0."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-31 -- 2008-08-15"


from .output import ufl_assert, ufl_warning
from .base import UFLObject, Terminal, as_ufl
from .indexing import Index, MultiIndex, DefaultDim, free_index_dimensions

# TODO: This ListVector/ListMatrix structure can probably be generalized to tensors.

class ListVector(UFLObject):
    __slots__ = ("_expressions", "_free_indices")
    
    def __init__(self, *expressions):
        expressions = [as_ufl(e) for e in expressions]
        ufl_assert(all(e.shape() == () for e in expressions), "Expecting scalar valued expressions.")
        
        self._free_indices = expressions[0].free_indices()
        self._expressions  = expressions
        
        ufl_assert(all(len(set(self._free_indices) ^ set(e.free_indices())) == 0 for e in expressions), \
            "Can't handle list of expressions with different free indices.")
        #ufl_assert(len(expressions.free_indices()) == 0, "Can't handle list of expressions with free indices.")
    
    def operands(self):
        return tuple(self._expressions)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return (len(self._expressions),)
    
    def __str__(self):
        return "<%s>" % ", ".join(str(e) for e in self._expressions)
    
    def __repr__(self):
        return "ListVector(*%s)" % repr(self._expressions)


class ListMatrix(UFLObject):
    __slots__ = ("_rowexpressions", "_free_indices", "_shape")
    
    def __init__(self, *rowexpressions):
        ufl_assert(all(isinstance(e, ListVector) for e in rowexpressions), \
            "Expecting list of rowexpressions.")
        
        self._rowexpressions = rowexpressions
        r = len(rowexpressions)
        c = rowexpressions[0].shape()[0]
        self._shape = (r, c)
        eset = set(rowexpressions[0].free_indices())
        
        for row in rowexpressions:
            ufl_assert(row.shape()[0] == c, "Inconsistent row size.")
            ufl_assert(all(e.shape() == () for e in row._expressions), \
                "Expecting scalar valued expressions.")
            ufl_assert(all(len(eset ^ set(e.free_indices())) == 0 for e in row._expressions), \
                "Can't handle list of expressions with different free indices.")
        #ufl_assert(len(expressions.free_indices()) == 0, "Can't handle list of expressions with free indices.")
    
    def operands(self):
        return tuple(self._rowexpressions)
    
    def free_indices(self):
        return self._rowexpressions[0].free_indices()
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        rowstrings = []
        for row in self._rowexpressions:
            rowstrings.append( ("[%s]" % ", ".join(str(e) for e in row._expressions)) ) 
        return "[ %s ]" % ", ".join(rowstrings)
    
    def __repr__(self):
        return "ListMatrix(*%s)" % repr(self._rowexpressions)


class Tensor(UFLObject):
    __slots__ = ("_expression", "_indices", "_free_indices", "_shape")
    
    def __init__(self, expression, indices):
        ufl_assert(isinstance(expression, UFLObject), "Expecting ufl expression.")
        ufl_assert(expression.shape() == (), "Expecting scalar valued expression.")
        self._expression = expression
        
        if isinstance(indices, MultiIndex): # if constructed from repr
            self._indices = indices
        else:
            # FIXME: I'm not sure about this:
            self._indices = MultiIndex(indices, len(expression.free_indices()))
        
        # Allowing Axis or FixedIndex here would make no sense
        ufl_assert(all(isinstance(i, Index) for i in self._indices._indices))
        
        eset = set(expression.free_indices())
        iset = set(self._indices._indices)
        jset = iset - eset
        self._free_indices = tuple(jset)
        ufl_assert(len(jset) == 0, "Missing indices %s in expression %s." % (jset, expression))
        
        dims = free_index_dimensions(expression)
        self._shape = tuple(dims[i] for i in self._indices._indices)
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "[Rank %d tensor A, such that A_{%s} = %s]" % (self.rank(), self._indices, self._expression)
    
    def __repr__(self):
        return "Tensor(%r, %r)" % (self._expression, self._indices)


def Vector(expressions, index = None):
    if index is None:
        ufl_assert(isinstance(expressions, (list, tuple)))
        return ListVector(*expressions)
    return Tensor(expressions, (index,))


def Matrix(expressions, indices = None):
    if indices is None:
        ufl_assert(isinstance(expressions, (list, tuple)))
        return ListMatrix(*[ListVector(*e) for e in expressions])
    return Tensor(expressions, indices)

