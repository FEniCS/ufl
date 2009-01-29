"""This module defines the single index types and some internal index utilities."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2009-01-29"

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.common import Counted
from ufl.terminal import Terminal

#--- Index types ---

# TODO: Should we make IndexBase a Terminal? The IndexSum and SpatialDerivative can have an Index instead of a MultiIndex.
#class NewIndexBase(Terminal):
#    def __init__(self):
#        Terminal.__init__(self)
#    
#    def shape(self):
#        error("")
#    
#    def __hash__(self):
#        return hash(repr(self))

class IndexBase(object):
    def __init__(self):
        pass

class Index(IndexBase, Counted):
    __slots__ = ()
    _globalcount = 0
    def __init__(self, count = None):
        IndexBase.__init__(self)
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

class FixedIndex(IndexBase):
    __slots__ = ("_value",)
    
    def __init__(self, value):
        IndexBase.__init__(self)
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

class MultiIndex(Terminal):
    __slots__ = ("_indices",)
    
    def __init__(self, ii):
        Terminal.__init__(self)
        if isinstance(ii, int):
            ii = (FixedIndex(ii),)
        elif isinstance(ii, IndexBase):
            ii = (ii,)
        elif isinstance(ii, tuple):
            ii = tuple(as_index(j) for j in ii)
        else:
            error("Expecting tuple of UFL indices.")
        self._indices = ii
    
    def free_indices(self):
        # This reflects the fact that a MultiIndex isn't a tensor expression
        error("Calling free_indices on MultiIndex is an error.")
    
    def index_dimensions(self):
        error("Calling index_dimensions on MultiIndex is an error.")
    
    def shape(self):
        error("Calling shape on MultiIndex is an error.")
    
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

def as_index(i):
    if isinstance(i, IndexBase):
        return i
    elif isinstance(i, int):
        return FixedIndex(i)
    elif isinstance(i, IndexBase):
        return (i,)
    error("Invalid object %s to create index from." % repr(i))

def as_multi_index(i):
    if isinstance(i, MultiIndex):
        return i
    return MultiIndex(i)

def indices(n):
    "Return a tuple of n new Index objects."
    return tuple(Index() for _i in range(n))

# TODO: Fix imports everywhere else instead
from ufl.indexutils import complete_shape
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
