"""This module defines the single index types and some internal index utilities."""

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
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
#
# First added:  2008-03-14
# Last changed: 2011-06-22

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import counted_init, EmptyDict
from ufl.terminal import UtilityType
from itertools import izip

#--- Index types ---

class IndexBase(object):
    __slots__ = ()
    def __init__(self):
        pass

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

    def __eq__(self, other):
        return isinstance(other, Index) and (self._count == other._count)

    def __str__(self):
        c = str(self._count)
        if len(c) > 1:
            c = "{%s}" % c
        return "i_%s" % c

    def __repr__(self):
        return "Index(%d)" % self._count

    def __hash__(self):
        return hash(repr(self))

class FixedIndex(IndexBase):
    """UFL value: An index with a specific value assigned."""
    __slots__ = ("_value", "_hash")
    _cache = {}
    def __new__(cls, value):
        self = FixedIndex._cache.get(value)
        if self is None:
            if not isinstance(value, int):
                error("Expecting integer value for fixed index.")
            self = IndexBase.__new__(cls)
            FixedIndex._cache[value] = self
        return self

    def __getnewargs__(self):
        return (self._value,)

    def __init__(self, value):
        if not hasattr(self, "_value"):
            IndexBase.__init__(self)
            self._value = value
            self._hash = hash(repr(self))

    def __eq__(self, other):
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

    def __hash__(self):
        return self._hash

class MultiIndex(UtilityType):
    "Represents a sequence of indices, either fixed or free."
    __slots__ = ("_indices", "_idims",)

    _cache = {}
    def __new__(cls, ii, idims):
        # Convert ii to proper type and check if input is cache-able
        if isinstance(ii, FixedIndex):
            key = (ii._value,)
            ii = (ii,)
        elif isinstance(ii, Index):
            key = None # Not cachable
            ii = (ii,)
        elif isinstance(ii, int):
            key = (ii,)
            ii = (FixedIndex(ii),)
        elif isinstance(ii, tuple):
            ii = tuple(as_index(j) for j in ii)
            if all(isinstance(jj, FixedIndex) for jj in ii):
                key = tuple(jj._value for jj in ii)
            else:
                key = None
        else:
            error("Expecting tuple of UFL indices, not %s." % (ii,))

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

        # Initialize here to avoid repeating the checks on ii from above in __init__
        self._init(ii, idims)
        return self

    def __getnewargs__(self):
        return (self._indices, self._idims)

    def __init__(self, ii, idims):
        pass

    def _init(self, ii, idims):
        UtilityType.__init__(self)

        self._indices = ii
        self._idims = dict(idims) if idims else EmptyDict

        if any(not isinstance(idims.get(k,0), int) for k in ii if isinstance(k, Index)):
            error("Missing index or invalid dimension in provided idims.")

    def evaluate(self, x, mapping, component, index_values):
        # Build component from index values
        component = []
        for i in self._indices:
            if isinstance(i, FixedIndex):
                component.append(i._value)
            elif isinstance(i, Index):
                component.append(index_values[i])
        return tuple(component)

    def free_indices(self):
        return tuple(i for i in set(self._indices) if isinstance(i, Index))

    def index_dimensions(self):
        return self._idims

    def __add__(self, other):
        sid = self.index_dimensions()
        oid = other.index_dimensions()
        if sid or oid:
            idims = dict(sid)
            idims.update(oid)
        else:
            idims = EmptyDict
        if isinstance(other, tuple):
            return MultiIndex(self._indices + other, idims)
        elif isinstance(other, MultiIndex):
            return MultiIndex(self._indices + other._indices, idims)
        return NotImplemented

    def __radd__(self, other):
        sid = self.index_dimensions()
        oid = other.index_dimensions()
        if sid or oid:
            idims = dict(sid)
            idims.update(oid)
        else:
            idims = EmptyDict
        if isinstance(other, tuple):
            return MultiIndex(other + self._indices, idims)
        elif isinstance(other, MultiIndex):
            return MultiIndex(other._indices + self._indices, idims)
        return NotImplemented

    def __str__(self):
        return ", ".join(str(i) for i in self._indices)

    def __repr__(self):
        return "MultiIndex(%r, %r)" % (self._indices, self._idims)

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
    else:
        error("Invalid object %s to create index from." % repr(i))

def _make_idims(ii, shape):
    if shape is None:
        return None
    else:
        return dict((j,d) for (j,d) in izip(ii, shape) if isinstance(j, Index))

def as_multi_index(ii, shape=None):
    if isinstance(ii, MultiIndex):
        if ii._idims:
            return ii
        else:
            ii = ii._indices
    elif not isinstance(ii, tuple):
        ii = (ii,)
    return MultiIndex(ii, _make_idims(ii, shape))

def indices(n):
    "UFL value: Return a tuple of n new Index objects."
    return tuple(Index() for _i in range(n))

# TODO: Fix imports everywhere else instead
from ufl.indexutils import complete_shape
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum

