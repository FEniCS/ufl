# -*- coding: utf-8 -*-
"""This module defines the single index types and some internal index utilities."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs and Anders Logg
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
# Modified by Massimiliano Leoni, 2016.

import six
from six.moves import xrange as range

from ufl.utils.py23 import as_native_str
from ufl.utils.py23 import as_native_strings
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.utils.counted import counted_init
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal

# Export list for ufl.classes
__all_classes__ = ["IndexBase", "FixedIndex", "Index"]


class IndexBase(object):
    """Base class for all indices."""
    __slots__ = ()

    def __init__(self):
        pass

    def __unicode__(self):
        # Only in python 2
        return str(self).decode("utf-8")


# @six.python_2_unicode_compatible
class FixedIndex(IndexBase):
    """UFL value: An index with a specific value assigned."""
    __slots__ = as_native_strings(("_value", "_hash"))

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
        return isinstance(other, FixedIndex) and (self._value == other._value)

    def __int__(self):
        return self._value

    def __str__(self):
        return "%d" % self._value

    def __repr__(self):
        r = "FixedIndex(%d)" % self._value
        return as_native_str(r)


# @six.python_2_unicode_compatible
class Index(IndexBase):
    """UFL value: An index with no value assigned.

    Used to represent free indices in Einstein indexing notation."""
    __slots__ = as_native_strings(("_count",))

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
        c = str(self._count)
        if len(c) > 1:
            c = "{%s}" % c
        return "i_%s" % c

    def __repr__(self):
        r = "Index(%d)" % self._count
        return as_native_str(r)


@ufl_type()
class MultiIndex(Terminal):
    "Represents a sequence of indices, either fixed or free."
    __slots__ = as_native_strings(("_indices",))

    _cache = {}

    def __getnewargs__(self):
        return (self._indices,)

    def __new__(cls, indices):
        ufl_assert(isinstance(indices, tuple), "Expecting a tuple of indices.")

        if all(isinstance(ind, FixedIndex) for ind in indices):
            # Cache multiindices consisting of purely fixed indices (aka flyweight pattern)
            key = tuple(ind._value for ind in indices)
            self = MultiIndex._cache.get(key)
            if self is not None:
                return self
            self = Terminal.__new__(cls)
            MultiIndex._cache[key] = self
        else:
            # Create a new object if we have any free indices (too many combinations to cache)
            ufl_assert(all(isinstance(ind, IndexBase) for ind in indices), "Expecting only Index and FixedIndex objects.")
            self = Terminal.__new__(cls)

        # Initialize here instead of in __init__ to avoid overwriting self._indices from cached objects
        self._init(indices)
        return self

    def __init__(self, indices):
        pass

    def _init(self, indices):
        Terminal.__init__(self)
        self._indices = indices

    def indices(self):
        "Return tuple of indices."
        return self._indices

    def _ufl_compute_hash_(self):
        return hash(("MultiIndex",) + tuple(hash(ind) for ind in self._indices))

    def __eq__(self, other):
        return isinstance(other, MultiIndex) and \
            self._indices == other._indices

    def evaluate(self, x, mapping, component, index_values):
        "Evaluate index."
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
        "This shall not be used."
        error("Multiindex has no shape (it is not a tensor expression).")

    @property
    def ufl_free_indices(self):
        "This shall not be used."
        error("Multiindex has no free indices (it is not a tensor expression).")

    @property
    def ufl_index_dimensions(self):
        "This shall not be used."
        error("Multiindex has no free indices (it is not a tensor expression).")

    def is_cellwise_constant(self):
        "Always True."
        return True

    def ufl_domains(self):
        "Return tuple of domains related to this terminal object."
        return ()

    # --- Adding multiindices ---

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

    # --- String formatting ---

    def __str__(self):
        return ", ".join(str(i) for i in self._indices)

    def __repr__(self):
        r = "MultiIndex(%r)" % (self._indices,)
        return as_native_str(r)

    # --- Iteration protocol ---

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._indices[i]

    def __iter__(self):
        return iter(self._indices)


def as_multi_index(ii, shape=None):
    "Return a ``MultiIndex`` version of *ii*."
    if isinstance(ii, MultiIndex):
        return ii
    elif not isinstance(ii, tuple):
        ii = (ii,)
    return MultiIndex(ii)


def indices(n):
    "UFL value: Return a tuple of :math:`n` new Index objects."
    return tuple(Index() for i in range(n))
