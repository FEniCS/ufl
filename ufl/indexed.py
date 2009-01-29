"""This module defines the Indexed class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-01-28"

from ufl.expr import Expr
from ufl.indexing import IndexBase, Index, FixedIndex, MultiIndex
from ufl.assertions import ufl_assert, assert_expr, assert_instance

#--- Indexed expression ---


def build_unique_indices(operands, multiindex=None, shape=None):
    "Build tuple of unique indices, including repeated ones."
    s = set()
    fi = []
    idims = {}
    for o in operands:
        if isinstance(o, MultiIndex):
            # TODO: This introduces None, better way? 
            ofi = o._indices
            oid = dict((i, None) for i in o) 
            #if shape is None:
            #    shape = (None,)*len(o)
            #oid = dict((i, shape[j]) for (j, i) in enumerate(ofi))
        else:
            ofi = o.free_indices()
            oid = o.index_dimensions()
        
        for i in ofi:
            if i in s:
                ri.append(i)
            else:
                fi.append(i)
                idims[i] = oid[i]
                s.add(i)
    return fi, ri, idims



class Indexed(Expr):
    def __init__(self, A, ii):
        fi, ri, idims = build_unique_indices((A,), ii, A.shape())
        self._fi = fi
        self._idims = idims
    
    def free_indices(self):
        return self._fi
    
    def index_dimensions(self):
        return self._idims



def extract_indices_for_indexed(indices, shape):
    """Analyse a tuple of indices and a shape tuple,
    and return a 4-tuple with the following information:
    
    @param shape
        New shape tuple after applying indices to given shape.
    @param free_indices
        Tuple of unique indices with no value
        (Index, no implicit summation)
    @param repeated_indices
        Tuple of indices that occur twice
        (Index, implicit summation)
    @param index_dimensions
        Dictionary (Index: int) with dimensions of each Index,
        taken from corresponding positions in shape.
    """
    # Validate input
    assert_instance(indices, tuple)
    assert_instance(shape, tuple)
    ufl_assert(all(isinstance(i, IndexBase) for i in indices), \
        "Expecting objects of type Index or FixedIndex, not %s." % repr(indices))
    ufl_assert(len(shape) == len(indices), "Expecting tuples of equal length.")
    
    # Get index dimensions from shape
    index_dimensions = dict((idx, dim) for (idx, dim) in zip(indices, shape)
                            if isinstance(idx, Index))
    
    # Count repetitions of indices
    index_count = defaultdict(int)
    for idx in indices:
        if isinstance(idx, Index):
            index_count[idx] += 1
    ufl_assert(all(i <= 2 for i in index_count.values()),
               "Too many index repetitions in %s" % repr(indices))
    
    # Split indices based on repetition count
    free_indices     = tuple(idx for idx in indices
                             if index_count[idx] == 1)
    repeated_indices = tuple(idx for idx in index_count.keys()
                             if index_count[idx] == 2)
    
    # Consistency check
    fixed_indices = tuple(idx for idx in indices 
                          if isinstance(idx, FixedIndex))
    n = len(fixed_indices) + len(free_indices) + 2*len(repeated_indices)
    ufl_assert(n == len(indices),
               "Logic breach in extract_indices_for_indexed.")
    
    return (free_indices, repeated_indices, index_dimensions)

#--- Indexed expression ---

class Indexed(Expr):
    __slots__ = ("_expression", "_indices",
                 "_free_indices", "_index_dimensions")
    def __init__(self, expression, indices):
        Expr.__init__(self)
        self._expression = expression
        
        if not isinstance(indices, MultiIndex):
            # unless constructed from repr
            indices = MultiIndex(indices)
        self._indices = indices
        
        msg = "Invalid number of indices (%d) for tensor "\
            "expression of rank %d:\n\t%r\n" % \
            (len(self._indices), expression.rank(), expression)
        ufl_assert(expression.rank() == len(self._indices), msg)
        

        shape = expression.shape()
        f, r, d = extract_indices_for_indexed(self._indices._indices, shape) # FIXME: Replace this code
        # Find additional free and repeated indices from expression # TODO: Merge with extract_indices_for_indexed?
        efi = expression.free_indices()
        eid = expression.index_dimensions()
        fi = list(f)
        ri = list(r)
        for i in efi:
            if i in f:
                ri.append(i)
                fi.remove(i) # FIXME: Don't remove anymore
            else:
                fi.append(i)
                d[i] = eid[i]
        
        self._free_indices = tuple(fi)
        self._index_dimensions = d
    
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

