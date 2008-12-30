"""This module defines the single index types and some internal index utilities."""


__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-11-26"

from collections import defaultdict
from ufl.output import ufl_assert, ufl_warning, ufl_error
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.common import Counted

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
        return "i_{%d}" % self._count
    
    def __repr__(self):
        return "Index(%d)" % self._count

def indices(n):
    "Return a tuple of n new Index objects."
    return tuple(Index() for i in range(n))

def relabel(A, indexmap):
    "Relabel free indices of A with new indices, using the given mapping."
    ii = tuple(sorted(indexmap.keys()))
    jj = tuple(indexmap[i] for i in ii)
    ufl_assert(all(isinstance(i, Index) for i in ii), "Expecting Index objects.")
    ufl_assert(all(isinstance(j, Index) for j in jj), "Expecting Index objects.")
    return as_tensor(A, ii)[jj]

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

class AxisType(object):
    __slots__ = ()
    
    def __init__(self):
        pass
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return isinstance(other, AxisType)
    
    def __str__(self):
        return ":"
    
    def __repr__(self):
        return "Axis"

# Collect all index types to shorten isinstance(a, _indextypes)
_indextypes = (Index, FixedIndex, AxisType) # TODO: Use superclass instead

# Only need one of these, like None, Ellipsis etc., we can
# then use either "a is Axis" or "isinstance(a, AxisType)"
Axis = AxisType()

#--- Indexing ---

class MultiIndex(Terminal):
    __slots__ = ("_indices", "_rank")
    
    def __init__(self, indices, rank):
        Terminal.__init__(self)
        self._indices = as_index_tuple(indices, rank)
        self._rank = rank
        ufl_assert(len(self._indices) == rank, "No? Why?")
    
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
        return "MultiIndex(%r, %d)" % (self._indices, self._rank)
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, i):
        return self._indices[i]
    
    def __iter__(self):
        return iter(self._indices)
    
    def __eq__(self, other):
        return isinstance(other, MultiIndex) and \
            self._indices == other._indices and \
            self._rank == other._rank

class Indexed(Expr):
    __slots__ = ("_expression", "_indices",
                 "_free_indices", "_index_dimensions", "_shape",
                 "_repeated_indices",)
    def __init__(self, expression, indices):
        Expr.__init__(self)
        self._expression = expression
        
        if not isinstance(indices, MultiIndex):
            # if constructed from repr
            indices = MultiIndex(indices, expression.rank())
        self._indices = indices
        
        msg = "Invalid number of indices (%d) for tensor "\
            "expression of rank %d:\n\t%r\n" % \
            (len(self._indices), expression.rank(), expression)
        ufl_assert(expression.rank() == len(self._indices), msg)
        
        shape = expression.shape()
        f, r, s, d = extract_indices(self._indices._indices, shape)
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

    def __str__(self):
        return "%s[%s]" % (self._expression, self._indices)
    
    def __repr__(self):
        return "Indexed(%r, %r)" % (self._expression, self._indices)
    
    def __getitem__(self, key):
        ufl_error("Object is already indexed: %r" % self)

def as_index(i):
    """Takes something the user might input as part of an
    index tuple, and returns an actual UFL index object."""
    if isinstance(i, _indextypes):
        return i
    elif isinstance(i, int):
        return FixedIndex(i)
    elif isinstance(i, slice):
        ufl_assert(i == slice(None), "Partial slices not implemented, only [:]")
        return Axis
    else:
        ufl_error("Can't convert this object to index: %r" % i)

def as_index_tuple(indices, rank):
    """Takes something the user might input as an index tuple
    inside [], and returns a tuple of actual UFL index objects.
    
    These types are supported:
    - Index
    - int => FixedIndex
    - Complete slice (:) => Axis
    - Ellipsis (...) => multiple Axis
    """
    if not isinstance(indices, tuple):
        indices = (indices,)
    pre  = []
    post = []
    found = False
    for idx in indices:
        if idx == Ellipsis:
            ufl_assert(not found, "Found duplicate ellipsis.")
            found = True
        else:
            if not found:
                pre.append(as_index(idx))
            else:
                post.append(as_index(idx))
    
    # replace ellipsis with a number of Axis objects
    num_axis = rank - len(pre) - len(post)
    ufl_assert(num_axis >= 0, "Invalid number of indices "\
        "(%d) for given rank (%d)." % (len(indices), rank))
    indices = tuple(pre + [Axis]*num_axis + post)
    return indices

def extract_indices(indices, shape=None):
    """Analyse a tuple of indices, and return a
    3-tuple with the following information:
    
    @param free_indices
        Tuple of unique indices with no value
        (Index, no implicit summation)
    @param repeated_indices
        Tuple of indices that occur twice
        (Index, implicit summation)
    @param shape
        Tuple with the combined shape computed 
        from axes that have no associated index
    """
    ufl_assert(isinstance(indices, tuple), "Expecting index tuple.")
    ufl_assert(all(isinstance(i, _indextypes) for i in indices), \
        "Expecting objects of type Index, FixedIndex, or Axis.")
    if shape is not None:
        ufl_assert(isinstance(shape, tuple), "Expecting index tuple.")
        #ufl_assert(len(shape) == len(indices), "Expecting tuples of equal length.")
        ufl_assert(len(shape) <= len(indices),
            "Expecting at least as many indices as the shape is.")
    
    index_dimensions = {}
    index_count = defaultdict(int)
    for i,idx in enumerate(indices):
        if isinstance(idx, Index):
            index_count[idx] += 1
            if shape is not None:
                index_dimensions[idx] = shape[i]
    
    newshape = []
    if shape is not None:
        for i,idx in enumerate(indices):
            if isinstance(idx, AxisType):
                ufl_assert(i < len(shape), "Indexing logic is messed up.")
                newshape.append(shape[i])
    else:
        ufl_assert(not any(isinstance(i, AxisType) for i in indices), \
            "Not expecting Axis when shape is not specified.")
    newshape = tuple(newshape)
    
    ufl_assert(all(i <= 2 for i in index_count.values()),
               "Too many index repetitions in %s" % repr(indices))
    
    # Split based on count
    unique_indices = index_count.keys()
    free_indices     = tuple([i for i in unique_indices if index_count[i] == 1])
    repeated_indices = tuple([i for i in unique_indices if index_count[i] == 2])

    # Consistency check
    fixed_indices = [(i,idx) for (i,idx) in enumerate(indices) \
                     if isinstance(idx, FixedIndex)]
    ufl_assert(len(fixed_indices)+len(free_indices) + \
               2*len(repeated_indices)+len(newshape)== len(indices),\
               "Logic breach in extract_indices.")
 
    return (free_indices, repeated_indices, newshape, index_dimensions)

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
    fis = tuple(i for i in chain(ai,bi) if not i in ris)
    #n = len(ris) + len(fis)
    #ufl_assert(n == ?)
    return (fis, ris)

