"""This module defines the single index types and some internal index utilities."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2009-02-20"

from ufl.log import error
from ufl.common import Counted
from ufl.terminal import UtilityType

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
    __slots__ = ("_str", "_repr", "_hash")
    _globalcount = 0
    def __init__(self, count=None):
        IndexBase.__init__(self)
        Counted.__init__(self, count)

        c = str(self._count)
        if len(c) > 1:
            c = "{%s}" % c
        self._str = "i_%s" % c
        self._repr = "Index(%d)" % self._count
        self._hash = hash(self._repr)
    
    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Index) and (self._count == other._count)
    
    def __str__(self):
        return self._str
    
    def __repr__(self):
        return self._repr

class FixedIndex(IndexBase):
    __slots__ = ("_value", "_repr")
    
    def __init__(self, value):
        IndexBase.__init__(self)
        if not isinstance(value, int):
            error("Expecting integer value for fixed index.")
        self._value = value
        self._repr = "FixedIndex(%d)" % self._value
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        if isinstance(other, FixedIndex):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == other
        return False
    
    def __int__(self):
        return self._value
    
    def __str__(self):
        return "%d" % self._value
    
    def __repr__(self):
        return self._repr

class MultiIndex(UtilityType):
    __slots__ = ("_indices", "_str", "_repr")
    
    def __init__(self, ii):
        UtilityType.__init__(self)
        if isinstance(ii, int):
            ii = (FixedIndex(ii),)
        elif isinstance(ii, IndexBase):
            ii = (ii,)
        elif isinstance(ii, tuple):
            ii = tuple(as_index(j) for j in ii)
        else:
            error("Expecting tuple of UFL indices.")
        self._indices = ii
        self._str = ", ".join(str(i) for i in self._indices)
        self._repr = "MultiIndex(%r)" % (self._indices,)
    
    def evaluate(self, x, mapping, component, index_values):
        # Build component from index values
        component = []
        for i in self._indices:
            if isinstance(i, FixedIndex):
                component.append(i._value)
            elif isinstance(i, Index):
                component.append(index_values[i])
        return tuple(component)

    def __add__(self, other):
        if isinstance(other, tuple):
            return MultiIndex(self._indices + other)
        elif isinstance(other, MultiIndex):
            return MultiIndex(self._indices + other._indices)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, tuple):
            return MultiIndex(other + self._indices)
        elif isinstance(other, MultiIndex):
            return MultiIndex(other._indices + self._indices)
        return NotImplemented

    def __str__(self):
        return self._str
    
    def __repr__(self):
        return self._repr
    
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
