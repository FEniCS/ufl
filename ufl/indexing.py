"""This module defines the single index types and some internal index utilities."""

# Copyright (C) 2008-2011 Martin Sandve Alnes and Anders Logg
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
from ufl.common import Counted
from ufl.terminal import UtilityType

#--- Index types ---

class IndexBase(object):
    def __init__(self):
        pass

class Index(IndexBase, Counted):
    """UFL value: An index with no value assigned.

    Used to represent free indices in Einstein indexing notation."""
    __slots__ = ("_str", "_repr", "_hash")
    _globalcount = 0
    def __init__(self, count=None):
        IndexBase.__init__(self)
        Counted.__init__(self, count)

        c = str(self._count)
        if len(c) > 1:
            c = "{%s}" % c
        self._str = "i_%s" % c
        self._repr = "Index(%d)" % self._count # REPR: cache or not?
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
    """UFL value: An index with a specific value assigned."""
    __slots__ = ("_value", "_repr")
    def __init__(self, value):
        IndexBase.__init__(self)
        if not isinstance(value, int):
            error("Expecting integer value for fixed index.")
        self._value = value
        self._repr = "FixedIndex(%d)" % self._value # REPR: cache or not?
    
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

_fixed_indices = {}
def fixed_index(value): # TODO: move into a FixedIndex.__new__ implementation
    ii = _fixed_indices.get(value)
    if ii is None:
        ii = FixedIndex(value)
        _fixed_indices[value] = ii
    return ii

class MultiIndex(UtilityType):
    "Represents a sequence of indices, either fixed or free."
    __slots__ = ("_indices",)

    def __init__(self, ii, idims=None):
        UtilityType.__init__(self)
        
        if isinstance(ii, int):
            ii = (fixed_index(ii),)
        elif isinstance(ii, IndexBase):
            ii = (ii,)
        elif isinstance(ii, tuple):
            ii = tuple(as_index(j) for j in ii)
        else:
            error("Expecting tuple of UFL indices.")

        # TODO: Remove "idims is None" when it can no longer occur
        if idims is None:
            warning("No index dimensions provided in MultiIndex.")
            #error("No index dimensions provided in MultiIndex.")
        else:
            idims = dict(idims)
            for k in ii:
                if isinstance(k, Index):
                    if not k in idims:
                        error("Missing index in the provided idims.")
                    else:
                        ufl_assert(isinstance(idims[k], int),
                                   "Non-integer index dimension provided.")
        self._indices = ii
        self._idims = idims
    
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
        if self._idims is None:
            error("No index dimensions were provided for this multiindex.")
        return self._idims

    def __add__(self, other):
        idims = dict(self.index_dimensions())
        idims.update(other.index_dimensions())
        if isinstance(other, tuple):
            return MultiIndex(self._indices + other, idims)
        elif isinstance(other, MultiIndex):
            return MultiIndex(self._indices + other._indices, idims)
        return NotImplemented

    def __radd__(self, other):
        idims = dict(self.index_dimensions())
        idims.update(other.index_dimensions())
        if isinstance(other, tuple):
            return MultiIndex(other + self._indices, idims)
        elif isinstance(other, MultiIndex):
            return MultiIndex(other._indices + self._indices, idims)
        return NotImplemented

    def __str__(self):
        return ", ".join(str(i) for i in self._indices)

    def __repr__(self):
        return "MultiIndex(%r, %r)" % (self._indices, self._idims) # REPR: cache or not?

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
        return fixed_index(i)
    elif isinstance(i, IndexBase):
        return (i,)
    error("Invalid object %s to create index from." % repr(i))

def _make_idims(ii, shape):
    if shape is None:
        return None
    else:
        return dict((j,d) for (j,d) in zip(ii, shape) if isinstance(j, Index))

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
