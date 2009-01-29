"""This module defines the Indexed class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-01-29"

from collections import defaultdict
from ufl.log import error
from ufl.expr import Expr
from ufl.indexing import IndexBase, Index, FixedIndex, MultiIndex, as_multi_index
from ufl.indexutils import unique_indices
from ufl.assertions import ufl_assert, assert_expr, assert_instance

#--- Indexed expression ---

class Indexed(Expr):
    __slots__ = ("_expression", "_indices",
                 "_free_indices", "_index_dimensions")
    def __init__(self, expression, indices):
        Expr.__init__(self)
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
        sh = A.shape()
        ri = self.repeated_indices() # FIXME: Rewrite below code, no implicit sums needed
        
        # Build component from indices
        subcomp = []
        ri_pos = defaultdict(tuple)
        for k, i in enumerate(ii):
            if isinstance(i, FixedIndex):
                subcomp.append(i._value)
            elif isinstance(i, Index):
                if i in ri:
                    # Postphone assignment of component item to repeated index summation
                    subcomp.append(None)
                    ri_pos[i] += (k,)
                else:
                    subcomp.append(index_values[i])
        
        # Handle implicit sums over repeated indices if necessary
        if ri:
            if len(ri) > 1: # TODO: Implement to allow A[i,i,j,j] etc
                error("TODO: Multiple repeated indices not implemented yet."\
                    " Note that A[i,i,j,j] = A[i,i,:,:][j,j].")
            
            # Get summation range
            idx, = ri # TODO: Only one! Need permutations to do more.
            if len(ri_pos[idx]) == 2:
                i0, i1 = ri_pos[idx]
                ufl_assert(sh[i0] == sh[i1], "Dimension mismatch in implicit sum over Indexed object.")
            else:
                i0, = ri_pos[idx]
                i1 = idx
            dim = sh[i0]
            
            # Accumulate values
            result = 0
            #for jj in permutations: # TODO: Only one! Need permutations to do more.
            for j in range(dim):
                #for ii in ri:
                #    i0, i1 = ri_pos[ii]
                #    subcomp[i0] = jj[...]
                #    subcomp[i1] = jj[...]
                subcomp[i0] = j
                if isinstance(i1, int):
                    subcomp[i1] = j
                    pushed = False
                else: # isinstance(i1, Index):
                    pushed = True
                    index_values.push(i1, j)
                result += A.evaluate(x, mapping, tuple(subcomp), index_values)
                if pushed:
                    index_values.pop()
        else:
            # No repeated indices makes this simple
            result = A.evaluate(x, mapping, tuple(subcomp), index_values)
        
        return result

    def __str__(self):
        return "%s[%s]" % (self._expression, self._indices)
    
    def __repr__(self):
        return "Indexed(%r, %r)" % (self._expression, self._indices)
    
    def __getitem__(self, key):
        error("Attempting to index with %r, but object is already indexed: %r" % (key, self))

