"""This module defines the single index types and some internal index utilities."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-08-18"

# Python imports
from collections import defaultdict

# UFL imports
from .output import ufl_assert, ufl_warning, ufl_error
from .base import UFLObject, Terminal
from .common import Counted

#--- Indexing ---

class Index(Counted):
    __slots__ = ()
    _globalcount = 0
    def __init__(self, count = None):
        Counted.__init__(self, count)
    
    def __str__(self):
        return "i_{%d}" % self._count
    
    def __repr__(self):
        return "Index(%d)" % self._count

class FixedIndex(object):
    __slots__ = ("_value",)
    
    def __init__(self, value):
        ufl_assert(isinstance(value, int), "Expecting integer value for fixed index.")
        self._value = value
    
    def __str__(self):
        return "%d" % self._value
    
    def __repr__(self):
        return "FixedIndex(%d)" % self._value

class AxisType(object):
    __slots__ = ()
    
    def __init__(self):
        pass
    
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
    __slots__ = ("_indices",)
    
    def __init__(self, indices, rank):
        self._indices = as_index_tuple(indices, rank)
    
    def free_indices(self):
        ufl_error("Calling free_indices on MultiIndex is undefined.")
    
    def shape(self):
        ufl_error("MultiIndex has no shape.")
    
    def __str__(self):
        return ", ".join(str(i) for i in self._indices)
    
    def __repr__(self):
        return "MultiIndex(%r, %d)" % (self._indices, len(self._indices))
    
    def __len__(self):
        return len(self._indices)

class Indexed(UFLObject):
    __slots__ = ("_expression", "_indices", "_fixed_indices",
                 "_free_indices", "_repeated_indices", "_rank", "_shape")
    
    def __init__(self, expression, indices):
        self._expression = expression
        
        if isinstance(indices, MultiIndex): # if constructed from repr
            self._indices = indices
        else:
            self._indices = MultiIndex(indices, expression.rank())
        
        msg = "Invalid number of indices (%d) for tensor expression of rank %d:\n\t%r\n" % \
            (len(self._indices), expression.rank(), expression)
        ufl_assert(expression.rank() == len(self._indices), msg)
        
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = \
            extract_indices(self._indices._indices)
        
        # TODO: We don't need to store all these here, remove the ones we
        #       don't use after implementing summation expansion.
        self._fixed_indices      = fixed_indices
        self._free_indices       = free_indices
        self._repeated_indices   = repeated_indices
        self._rank = num_unassigned_indices
        
        s = expression.shape()
        idx = self._indices._indices
        self._shape = tuple(s[i] for i in range(len(idx)) if isinstance(idx[i], AxisType))
        ufl_assert(self._rank == len(self._shape),
            "Logic breach in Indexed.__init__, rank is %d and shape is %r" % (self._rank, self._shape))
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
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

def extract_indices(indices):
    """Analyse a tuple of indices, and return a 4-tuple with the following information:
    - fixed_indices = tuple of indices with a constant value (FixedIndex)
    - free_indices = tuple of unique indices with no value (Index, no implicit summation)
    - repeated_indices = tuple of indices that occur twice (Index, implicit summation)
    - num_unassigned_indices = int, number of axes in that have no associated index
    """
    ufl_assert(isinstance(indices, tuple), "Expecting index tuple.")
    ufl_assert(all(isinstance(i, _indextypes) for i in indices), "Expecting proper UFL objects.")

    fixed_indices = [(i,idx) for i,idx in enumerate(indices) if isinstance(idx, FixedIndex)]
    num_unassigned_indices = sum(1 for i in indices if i is Axis)

    index_count = defaultdict(int)
    for i in indices:
        if isinstance(i, Index):
            index_count[i] += 1
    
    unique_indices = index_count.keys()
    
    ufl_assert(all(i <= 2 for i in index_count.values()),
               "Too many index repetitions in %s" % repr(indices))
    free_indices     = [i for i in unique_indices if index_count[i] == 1]
    repeated_indices = [i for i in unique_indices if index_count[i] == 2]

    # use tuples for consistency
    fixed_indices    = tuple(fixed_indices)
    free_indices     = tuple(free_indices)
    repeated_indices = tuple(repeated_indices)
    
    ufl_assert(len(fixed_indices)+len(free_indices) + \
               2*len(repeated_indices)+num_unassigned_indices == len(indices),\
               "Logic breach in extract_indices.")
    
    return (fixed_indices, free_indices,
            repeated_indices, num_unassigned_indices)


class UnassignedDimType(object):
    __slots__ = ()
    
    def __init__(self):
        pass
    
    def __str__(self):
        return "?"
    
    def __repr__(self):
        return "UnassignedDimType()"

UnassignedDim = UnassignedDimType()


def compare_shapes(a, b):
    if len(a) != len(b):
        return False
    return all(((i == j) or isinstance(i, UnassignedDimType) or \
        isinstance(j, UnassignedDimType)) for (i,j) in zip(a,b))

def free_index_dimensions(e):
    # FIXME: Get the dimensions from the expression!
    ufl_warning("free_index_dimensions just returns UnassignedDim for everything, needs better implementation.")
    return dict((i, UnassignedDim) for i in e.free_indices())

