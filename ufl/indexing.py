"""This module defines the single index types and some internal index utilities."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-10-21"

# Python imports
from collections import defaultdict

# UFL imports
from .output import ufl_assert, ufl_warning, ufl_error
from .base import Expr, Terminal
from .common import Counted

#--- Indexing ---

class Index(Counted):
    __slots__ = ()
    _globalcount = 0
    def __init__(self, count = None):
        Counted.__init__(self, count)
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        if isinstance(other, Index):
            return self._count == other._count
        return False
    
    def __str__(self):
        return "i_{%d}" % self._count
    
    def __repr__(self):
        return "Index(%d)" % self._count

class FixedIndex(object):
    __slots__ = ("_value",)
    
    def __init__(self, value):
        ufl_assert(isinstance(value, int), "Expecting integer value for fixed index.")
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
_indextypes = (Index, FixedIndex, AxisType)

# Only need one of these, like None, Ellipsis etc., can use "a is Axis" or "isinstance(a, AxisType)"
Axis = AxisType()


#--- Indexing ---

class MultiIndex(Terminal):
    __slots__ = ("_indices", "_rank")
    
    def __init__(self, indices, rank):
        self._indices = as_index_tuple(indices, rank)
        self._rank = rank
        ufl_assert(len(self._indices) == rank, "No? Why?")
    
    def free_indices(self):
        ufl_error("Calling free_indices on MultiIndex is undefined.")
    
    def shape(self):
        ufl_error("MultiIndex has no shape.")
    
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
                 "_free_indices", "_shape",
                 "_repeated_indices",)
    def __init__(self, expression, indices):
        self._expression = expression
        
        if not isinstance(indices, MultiIndex):
            # if constructed from repr
            indices = MultiIndex(indices, expression.rank())
        self._indices = indices
        
        msg = "Invalid number of indices (%d) for tensor expression of rank %d:\n\t%r\n" % \
            (len(self._indices), expression.rank(), expression)
        ufl_assert(expression.rank() == len(self._indices), msg)
        
        shape = expression.shape()
        (self._free_indices, self._repeated_indices, self._shape) = \
            extract_indices(self._indices._indices, shape)
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices

    def repeated_indices(self):
        return self._repeated_indices
    
    def shape(self):
        return self._shape

    def repeated_index_dimensions(self, default_dim):
        d = {}
        shape = self._expression.shape()
        for k, i in enumerate(self._indices._indices):
            if i in self._repeated_indices:
                j = shape[k]
                d[i] = default_dim if isinstance(j, DefaultDimType) else j
        return d

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
        ufl_error("Can convert this object to index: %r" % i)

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
    ufl_assert(num_axis >= 0, "Invalid number of indices (%d) for given rank (%d)." % (len(indices), rank))
    indices = tuple(pre + [Axis]*num_axis + post)
    return indices


def extract_indices(indices, shape=None):
    """Analyse a tuple of indices, and return a 3-tuple with the following information:
    
    @param free_indices
        Tuple of unique indices with no value (Index, no implicit summation)
    @param repeated_indices
        Tuple of indices that occur twice (Index, implicit summation)
    @param shape
        Tuple with the combined shape computed from axes in that have no associated index
    """
    ufl_assert(isinstance(indices, tuple), "Expecting index tuple.")
    ufl_assert(all(isinstance(i, _indextypes) for i in indices), \
        "Expecting objects of type Index, FixedIndex, or Axis.")
    if shape is not None:
        ufl_assert(isinstance(shape, tuple), "Expecting index tuple.")
        #ufl_assert(len(shape) == len(indices), "Expecting tuples of equal length.")
        ufl_assert(len(shape) <= len(indices), "Expecting at least as many indices as the shape is.")
    
    index_count = defaultdict(int)
    for i,idx in enumerate(indices):
        if isinstance(idx, Index):
            index_count[idx] += 1
    
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
    fixed_indices = [(i,idx) for (i,idx) in enumerate(indices) if isinstance(idx, FixedIndex)]
    ufl_assert(len(fixed_indices)+len(free_indices) + \
               2*len(repeated_indices)+len(newshape)== len(indices),\
               "Logic breach in extract_indices.")
    
    return (free_indices, repeated_indices, newshape)


class DefaultDimType(object):
    __slots__ = ()
    
    def __init__(self):
        pass
    
    def __str__(self):
        return "?"
    
    def __repr__(self):
        return "DefaultDimType()"

DefaultDim = DefaultDimType()

def complete_shape(a, dim): # FIXME: If we can get rid of DefaultDim, we can get rid of this...
    b = list(a)
    for i,x in enumerate(b):
        if isinstance(x, DefaultDimType):
            b[i] = dim
    return tuple(b)

def compare_shapes(a, b, dim=None):
    if len(a) != len(b):
        return False
    if dim is None:
        return all(((i == j) or isinstance(i, DefaultDimType) or \
            isinstance(j, DefaultDimType)) for (i,j) in zip(a,b))
    else:
        return all(((i == j) or \
                    (isinstance(i, DefaultDimType) and j == dim) or \
                    (isinstance(j, DefaultDimType) and i == dim)) \
                    for (i,j) in zip(a,b))

def indices(n):
    return tuple(Index() for i in range(n))

