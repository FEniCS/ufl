"""This module defines the single index types and some internal index utilities."""

# Copyright (C) 2008-2014 Martin Sandve Alnes and Anders Logg
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.

from six.moves import zip
from six.moves import xrange as range

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import counted_init, EmptyDict
from ufl.terminal import UtilityType
from ufl.core.ufl_type import ufl_type

#--- Index types ---

class IndexBase(object):
    __slots__ = ()
    def __init__(self):
        pass


class FixedIndex(IndexBase):
    """UFL value: An index with a specific value assigned."""
    __slots__ = ("_value", "_hash")

    _cache = {}

    def __getnewargs__(self):
        return (self._value,)

    def __new__(cls, value):
        self = FixedIndex._cache.get(value)
        if self is None:
            if not isinstance(value, int):
                error("Expecting integer value for fixed index.")
            self = IndexBase.__new__(cls)
            self._init(value)
            FixedIndex._cache[value] = self
        return self

    def _init(self, value):
        IndexBase.__init__(self)
        self._value = value
        self._hash = hash(("FixedIndex", self._value))

    def __init__(self, value):
        pass

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # FIXME: Disallow comparison with int. If the user wants that, int(index) == value does that.
        #return isinstance(other, FixedIndex) and (self._value == other._value)
        if isinstance(other, FixedIndex):
            return self._value == other._value
        elif isinstance(other, int): # Allow scalar comparison
            return self._value == other
        return False

    def __int__(self):
        return self._value

    def __str__(self):
        return "%d" % self._value

    def __repr__(self):
        return "FixedIndex(%d)" % self._value


class Index(IndexBase):
    """UFL value: An index with no value assigned.

    Used to represent free indices in Einstein indexing notation."""
    __slots__ = ("_count",)

    _globalcount = 0

    def __init__(self, count=None):
        IndexBase.__init__(self)
        counted_init(self, count, Index)

    def count(self):
        return self._count

    def __hash__(self):
        return hash(("Index", self._count))

    def __eq__(self, other):
        return isinstance(other, Index) and (self._count == other._count)

    def __str__(self):
        return "i%d" % self._count

    def __repr__(self):
        return "Index(%d)" % self._count


def _as_index(i):
    if isinstance(i, IndexBase):
        return i
    elif isinstance(i, int):
        return FixedIndex(i)
    else:
        error("Invalid object %s to create index from." % repr(i))


@ufl_type()
class MultiIndex(UtilityType):
    "Represents a sequence of indices, either fixed or free."
    __slots__ = ("_indices",)

    _cache = {}

    def __getnewargs__(self):
        return (self._indices,)

    def x__new__(cls, indices):
        # FIXME: Require indices to be a tuple of FixedIndex/Index objects already
        assert isinstance(indices, tuple)
        if all(isinstance(ind, FixedIndex) for ind in indices):
            # Cache multiindices consisting of purely fixed indices (aka flyweight pattern)
            key = tuple(ind._value for ind in indices)
            self = MultiIndex._cache.get(key)
            if self is not None:
                return self
            self = UtilityType.__new__(cls)
            MultiIndex._cache[key] = self
        else:
            # Create a new object if we have any free indices (too many combinations to cache)
            assert all(isinstance(ind, IndexBase) for ind in indices)
            self = UtilityType.__new__(cls)

        # Initialize here instead of in __init__ to avoid overwriting self._indices from cached objects
        self._init(indices)
        return self

    def __new__(cls, indices):

        # Convert indices to proper type and check if input is cache-able
        if isinstance(indices, FixedIndex):
            key = (indices._value,)
            indices = (indices,)
        elif isinstance(indices, Index):
            key = None # Not cachable
            indices = (indices,)
        elif isinstance(indices, int):
            key = (indices,)
            indices = (FixedIndex(indices),)
        elif isinstance(indices, tuple):
            indices = tuple(_as_index(j) for j in indices)
            if all(isinstance(jj, FixedIndex) for jj in indices):
                key = tuple(jj._value for jj in indices)
            else:
                key = None
        else:
            error("Expecting tuple of UFL indices, not %s." % (indices,))

        if key is not None:
            # Lookup in cache if we have a key
            self = MultiIndex._cache.get(key)
            if self is not None:
                return self
            self = UtilityType.__new__(cls)
            MultiIndex._cache[key] = self
        else:
            # Or skip cache for other cases
            self = UtilityType.__new__(cls)

        # Initialize here to avoid repeating the checks on indices from above in __init__
        self._init(indices)
        return self

    def __init__(self, indices):
        pass

    def _init(self, indices):
        UtilityType.__init__(self)
        self._indices = indices

    def indices(self):
        return self._indices

    def evaluate(self, x, mapping, component, index_values):
        # Build component from index values
        component = []
        for i in self._indices:
            if isinstance(i, FixedIndex):
                component.append(i._value)
            elif isinstance(i, Index):
                component.append(index_values[i])
        return tuple(component)

    @property
    def ufl_shape(self):
        # In the future we may wish to let multiindex have shape = (len(self),)
        # but then self[0] should return the index at position 0 and index
        # classes must become part of the Expr hierarchy.
        error("MultiIndex is not a tensor-valued expression.")

    @property
    def ufl_free_indices(self):
        error("MultiIndex is not a tensor-valued expression.")

    @property
    def ufl_index_dimensions(self):
        error("MultiIndex is not a tensor-valued expression.")

    def free_indices(self):
        error("MultiIndex is not a tensor-valued expression.")

    def index_dimensions(self):
        error("MultiIndex is not a tensor-valued expression.")

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

def as_multi_index(ii, shape=None):
    if isinstance(ii, MultiIndex):
        return ii
    elif not isinstance(ii, tuple):
        ii = (ii,)
    return MultiIndex(ii)

def indices(n):
    "UFL value: Return a tuple of n new Index objects."
    return tuple(Index() for i in range(n))

# TODO: Fix imports everywhere else instead
from ufl.indexutils import complete_shape
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
