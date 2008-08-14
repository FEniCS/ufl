"""This module defines the single index types and some internal index utilities."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-08-14"

# Python imports
from collections import defaultdict

# UFL imports
from .output import ufl_assert, ufl_error
from .base import UFLObject, Terminal

#--- Indexing ---

class Index(object):
    __slots__ = ("_count",)
    
    _globalcount = 0
    def __init__(self, count = None):
        if count is None:
            self._count = Index._globalcount
            Index._globalcount += 1
        else:
            self._count = count
            if count >= Index._globalcount:
                Index._globalcount = count + 1
    
    def free_indices(self):
        ufl_error("Why would you want to get the free indices of an Index? Please explain at ufl-dev@fenics.org...")
    
    def rank(self):
        ufl_error("Why would you want to get the rank of an Index? Please explain at ufl-dev@fenics.org...")
    
    def __str__(self):
        return "i_{%d}" % self._count
    
    def __repr__(self):
        return "Index(%d)" % self._count

# TODO: Replace this class with int? If we don't make
#       the index classes UFLObject subclasses again, there
#       might not be any point in having this class.
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
        return "AxisType()"

# Collect all index types to shorten isinstance(a, _indextypes)
_indextypes = (Index, FixedIndex, AxisType)

# Only need one of these, like None, Ellipsis etc., can use "a is Axis" or "isinstance(a, AxisType)"
Axis = AxisType()

#--- Indexing ---

class MultiIndex(Terminal): # TODO: If single indices are made Terminal subclasses, this should inherit UFLObject instead.
    __slots__ = ("_indices",)
    
    def __init__(self, indices, rank):
        self._indices = as_index_tuple(indices, rank)
    
    def operands(self):
        return ()
        #return self._indices # TODO: To return the indices here, they should be Terminal subclasses. Do we want that or not?
    
    def free_indices(self):
        ufl_error("Why would you want to get the free indices of a MultiIndex? Please explain at ufl-dev@fenics.org...")
    
    def rank(self):
        ufl_error("Why would you want to get the rank of a MultiIndex? Please explain at ufl-dev@fenics.org...")
    
    def __str__(self):
        return ", ".join(str(i) for i in self._indices)
    
    def __repr__(self):
        return "MultiIndex(%r, %d)" % (self._indices, len(self._indices))

    def __len__(self):
        return len(self._indices)

class Indexed(UFLObject):
    __slots__ = ("_expression", "_indices", "_fixed_indices", "_free_indices", "_repeated_indices", "_rank")
    
    def __init__(self, expression, indices):
        self._expression = expression
        
        if isinstance(indices, MultiIndex): # if constructed from repr
            self._indices = indices
        else:
            self._indices = MultiIndex(indices, expression.rank())
        
        msg = "Invalid number of indices (%d) for tensor expression of rank %d:\n\t%r\n" % (len(self._indices), expression.rank(), expression)
        ufl_assert(expression.rank() == len(self._indices), msg)
        
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = extract_indices(self._indices._indices)
        # FIXME: We don't need to store all these here, remove the ones we don't use after implementing summation expansion.
        self._fixed_indices      = fixed_indices
        self._free_indices       = free_indices
        self._repeated_indices   = repeated_indices
        self._rank = num_unassigned_indices
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self._rank
    
    def __str__(self):
        return "%s[%s]" % (self._expression, self._indices)
    
    def __repr__(self):
        return "Indexed(%r, %r)" % (self._expression, self._indices)
    
    def __getitem__(self, key):
        ufl_error("Object is already indexed: %r" % self)

# Extend UFLObject with indexing operator
def _getitem(self, key):
    return Indexed(self, key)
UFLObject.__getitem__ = _getitem

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
    for j, idx in enumerate(indices):
        if not found:
            if idx == Ellipsis:
                found = True
            else:
                pre.append(as_index(idx))
        else:
            ufl_assert(idx != Ellipsis, "Found duplicate ellipsis.")
            post.append(as_index(idx))
    
    # replace ellipsis with a number of Axis objects
    indices = pre + [Axis]*(rank-len(pre)-len(post)) + post
    return tuple(indices)

def extract_indices(indices):
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
