"""This module defines the single index types and some internal index utilities."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2009-01-10"

from itertools import chain
from collections import defaultdict
from ufl.output import ufl_assert, ufl_warning, ufl_error
from ufl.common import Counted
from ufl.expr import Expr
from ufl.terminal import Terminal

#--- Indexing ---

class Index(Counted):
    __slots__ = ()
    _globalcount = 0
    def __init__(self, count = None):
        Counted.__init__(self, count)
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return isinstance(other, Index) and (self._count == other._count)
    
    def __str__(self):
        c = str(self._count)
        if len(c) > 1:
            c = "{%s}" % c
        return "i_%s" % c
    
    def __repr__(self):
        return "Index(%d)" % self._count

def indices(n):
    "Return a tuple of n new Index objects."
    return tuple(Index() for _i in range(n))

class FixedIndex(object):
    __slots__ = ("_value",)
    
    def __init__(self, value):
        ufl_assert(isinstance(value, int),
            "Expecting integer value for fixed index.")
        self._value = value
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        if isinstance(other, FixedIndex):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == other
        return False
    
    def __str__(self):
        return "%d" % self._value
    
    def __repr__(self):
        return "FixedIndex(%d)" % self._value

# Collect all index types to shorten isinstance(a, _indextypes)
_indextypes = (Index, FixedIndex) # TODO: Use superclass instead? Index, FreeIndex(Index), FixedIndex(Index)

#--- Indexing ---

class MultiIndex(Terminal):
    __slots__ = ("_indices",)
    
    def __init__(self, ii):
        Terminal.__init__(self)
        self._indices, axes = as_index_tuple(ii)
        ufl_assert(axes == (), "Not expecting slices at this point.")
    
    def free_indices(self):
        # This reflects the fact that a MultiIndex isn't a tensor expression
        ufl_error("Calling free_indices on MultiIndex is an error.")
    
    def index_dimensions(self):
        ufl_error("Calling index_dimensions on MultiIndex is an error.")
    
    def shape(self):
        ufl_error("Calling shape on MultiIndex is an error.")
    
    def __str__(self):
        return ", ".join(str(i) for i in self._indices)
    
    def __repr__(self):
        return "MultiIndex(%r)" % (self._indices,)
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, i):
        return self._indices[i]
    
    def __iter__(self):
        return iter(self._indices)
    
    def __eq__(self, other):
        return isinstance(other, MultiIndex) and \
            self._indices == other._indices

class Indexed(Expr):
    __slots__ = ("_expression", "_indices",
                 "_shape",
                 "_free_indices", "_repeated_indices", "_index_dimensions")
    def __init__(self, expression, indices):
        Expr.__init__(self)
        self._expression = expression
        
        ufl_assert(expression.free_indices() == (), "Currently not accepting free indices in indexed expression.") # FIXME: Figure this out!!!
        
        if not isinstance(indices, MultiIndex):
            # unless constructed from repr
            indices = MultiIndex(indices)
        self._indices = indices
        
        msg = "Invalid number of indices (%d) for tensor "\
            "expression of rank %d:\n\t%r\n" % \
            (len(self._indices), expression.rank(), expression)
        ufl_assert(expression.rank() == len(self._indices), msg)
        
        shape = expression.shape()
        s, f, r, d = extract_indices_for_indexed(self._indices._indices, shape)
        self._free_indices = f
        self._repeated_indices = r
        self._shape = s
        self._index_dimensions = d
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices

    def repeated_indices(self):
        return self._repeated_indices

    def index_dimensions(self):
        # FIXME: Can we remove this now?
        #d = {}
        #shape = self._expression.shape()
        #for k, i in enumerate(self._indices._indices):
        #    if i in self._repeated_indices:
        #        d[i] = shape[k]
        #return d
        return self._index_dimensions
    
    def shape(self):
        return self._shape
    
    def evaluate(self, x, mapping, component, index_values):
        index_values = FIXME
        component = FIXME
        a = self._expression.evaluate(x, mapping, component, index_values)
        return a

    def __str__(self):
        return "%s[%s]" % (self._expression, self._indices)
    
    def __repr__(self):
        return "Indexed(%r, %r)" % (self._expression, self._indices)
    
    def __getitem__(self, key):
        ufl_error("Attempting to index with %r, but object is already indexed: %r" % (key, self))

def as_index_tuple(ii):
    """Takes something the user might input as an index tuple
    inside [], and returns a tuple of actual UFL index objects.
    
    These types are supported:
    - Index
    - int => FixedIndex
    - Complete slice (:) => Axis
    - Ellipsis (...) => multiple Axis
    """
    if not isinstance(ii, tuple):
        ii = (ii,)
    
    # Convert all indices to Index or FixedIndex objects.
    # If there is an ellipsis, split the indices into before and after.
    pre  = []
    post = []
    axes = set()
    found = False
    for i in ii:
        if i == Ellipsis:
            ufl_assert(not found, "Found duplicate ellipsis.")
            found = True
        else:
            if isinstance(i, int):
                idx = FixedIndex(i)
            elif isinstance(i, (FixedIndex, Index)):
                idx = i
            elif isinstance(i, slice):
                if i == slice(None):
                    idx = Index()
                    axes.add(idx)
                else:
                    # TODO: Use ListTensor for partial slices?
                    ufl_error("Partial slices not implemented, only [:]")
            else:
                ufl_error("Can't convert this object to index: %r" % i)
            
            if found:
                post.append(idx)
            else:
                pre.append(idx)
    
    # Handle ellipsis as a number of complete slices
    num_axis = len(ii) - len(pre) - len(post)
    ii = indices(num_axis)
    axes.update(ii)
    
    # Construct final tuples to return
    pre.extend(ii)
    pre.extend(post)
    ii = tuple(pre)
    axes = tuple(i for i in ii if i in axes)
    
    return ii, axes

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
    ufl_assert(isinstance(indices, tuple), "Expecting index tuple.")
    ufl_assert(all(isinstance(i, _indextypes) for i in indices), \
        "Expecting objects of type Index or FixedIndex, not %s." % repr(indices))
    ufl_assert(isinstance(shape, tuple), "Expecting index tuple.")
    ufl_assert(len(shape) == len(indices), "Expecting tuples of equal length.")
    
    # Get index dimensions from shape
    index_dimensions = dict((idx, dim) for (idx, dim) in zip(indices, shape)
                            if isinstance(idx, Index))
    
    # Build new shape
    #newshape = tuple(dim for (idx, dim) in zip(indices, shape)
    #                 if isinstance(idx, AxisType)) # TODO: This will always be empty now, so skip it
    newshape = ()
    
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
    ufl_assert(len(fixed_indices) + len(free_indices) + \
               2*len(repeated_indices) + len(newshape) == len(indices),
               "Logic breach in extract_indices_for_indexed.")
    
    return (newshape, free_indices, repeated_indices, index_dimensions)

def extract_indices_for_product(indices):
    """Analyse a tuple of indices, and return a
    2-tuple with the following information:
    
    @param free_indices
        Tuple of unique indices with no value
        (Index, no implicit summation)
    @param repeated_indices
        Tuple of indices that occur twice
        (Index, implicit summation)
    """
    # Validate input
    ufl_assert(isinstance(indices, tuple), "Expecting index tuple.")
    ufl_assert(all(isinstance(i, _indextypes) for i in indices), \
        "Expecting objects of type Index or FixedIndex.")
    
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
    ufl_assert(len(fixed_indices) + len(free_indices) + \
               2*len(repeated_indices) == len(indices),
               "Logic breach in extract_indices_for_product.")
    
    return free_indices, repeated_indices

def complete_shape(shape, default_dim):
    "Complete shape tuple by replacing non-integers with a default dimension."
    return tuple((s if isinstance(s, int) else default_dim) for s in shape)

# TODO: Use these to simplify index handling code?
def get_common_indices(a, b):
    ai = a.free_indices()
    bi = b.free_indices()
    cis = set(ai) ^ set(bi)
    return cis

def get_free_indices(a, b):
    ai = a.free_indices()
    bi = b.free_indices()
    cis = set(ai) ^ set(bi)
    return tuple(i for i in chain(ai,bi) if not i in cis)

# TODO: Use this to simplify index handling code?
def split_indices(a, b):
    ai = a.free_indices()
    bi = b.free_indices()
    ais = set(ai)
    bis = set(bi)
    ris = ais ^ bis
    fi  = tuple(i for i in chain(ai,bi) if not i in ris)
    ri  = tuple(i for i in chain(ai,bi) if     i in ris)
    #n = len(ri) + len(fi)
    #ufl_assert(n == ?)
    return (fi, ri)

