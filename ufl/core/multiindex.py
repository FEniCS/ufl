"""This module defines the single index types and some internal index utilities."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016.


from ufl.utils.counted import Counted
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal

# Export list for ufl.classes
__all_classes__ = ["IndexBase", "FixedIndex", "Index"]


class IndexBase(object):
    """Base class for all indices."""
    __slots__ = ()

    def __init__(self):
        """Initialise."""


class FixedIndex(IndexBase):
    """UFL value: An index with a specific value assigned."""
    __slots__ = ("_value", "_hash")

    _cache = {}

    def __getnewargs__(self):
        """Get new args."""
        return (self._value,)

    def __new__(cls, value):
        """Create new FixedIndex."""
        self = FixedIndex._cache.get(value)
        if self is None:
            if not isinstance(value, int):
                raise ValueError("Expecting integer value for fixed index.")
            self = IndexBase.__new__(cls)
            self._init(value)
            FixedIndex._cache[value] = self
        return self

    def _init(self, value):
        """Initialise."""
        IndexBase.__init__(self)
        self._value = value
        self._hash = hash(("FixedIndex", self._value))

    def __init__(self, value):
        """Initialise."""

    def __hash__(self):
        """Hash."""
        return self._hash

    def __eq__(self, other):
        """Check equality."""
        return isinstance(other, FixedIndex) and (self._value == other._value)

    def __int__(self):
        """Convert to int."""
        return self._value

    def __str__(self):
        """Represent with a string."""
        return f"{self._value}"

    def __repr__(self):
        """Return representation."""
        return f"FixedIndex({self._value})"


class Index(IndexBase, Counted):
    """UFL value: An index with no value assigned.

    Used to represent free indices in Einstein indexing notation.
    """

    __slots__ = ("_count", "_counted_class")

    def __init__(self, count=None):
        """Initialise."""
        IndexBase.__init__(self)
        Counted.__init__(self, count, Index)

    def __hash__(self):
        """Hash."""
        return hash(("Index", self._count))

    def __eq__(self, other):
        """Check equality."""
        return isinstance(other, Index) and (self._count == other._count)

    def __str__(self):
        """Represent as a string."""
        c = f"{self._count}"
        if len(c) > 1:
            c = f"{{{c}}}"
        return f"i_{c}"

    def __repr__(self):
        """Return representation."""
        return f"Index({self._count})"


@ufl_type()
class MultiIndex(Terminal):
    """Represents a sequence of indices, either fixed or free."""
    __slots__ = ("_indices",)

    _cache = {}

    def __getnewargs__(self):
        """Get new args."""
        return (self._indices,)

    def __new__(cls, indices):
        """Create new MultiIndex."""
        if not isinstance(indices, tuple):
            raise ValueError("Expecting a tuple of indices.")

        if all(isinstance(ind, FixedIndex) for ind in indices):
            # Cache multiindices consisting of purely fixed indices
            # (aka flyweight pattern)
            key = tuple(ind._value for ind in indices)
            self = MultiIndex._cache.get(key)
            if self is not None:
                return self
            self = Terminal.__new__(cls)
            MultiIndex._cache[key] = self
        else:
            # Create a new object if we have any free indices (too
            # many combinations to cache)
            if not all(isinstance(ind, IndexBase) for ind in indices):
                raise ValueError("Expecting only Index and FixedIndex objects.")
            self = Terminal.__new__(cls)

        # Initialize here instead of in __init__ to avoid overwriting
        # self._indices from cached objects
        self._init(indices)
        return self

    def __init__(self, indices):
        """Initialise."""

    def _init(self, indices):
        """Initialise."""
        Terminal.__init__(self)
        self._indices = indices

    def indices(self):
        """Return tuple of indices."""
        return self._indices

    def _ufl_compute_hash_(self):
        """Compute UFL hash."""
        return hash(("MultiIndex",) + tuple(hash(ind) for ind in self._indices))

    def __eq__(self, other):
        """Check equality."""
        return isinstance(other, MultiIndex) and \
            self._indices == other._indices

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate index."""
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
        """Get the UFL shape.

        This should not be used.
        """
        raise ValueError("Multiindex has no shape (it is not a tensor expression).")

    @property
    def ufl_free_indices(self):
        """Get the UFL free indices.

        This should not be used.
        """
        raise ValueError("Multiindex has no free indices (it is not a tensor expression).")

    @property
    def ufl_index_dimensions(self):
        """Get the UFL index dimensions.

        This should not be used.
        """
        raise ValueError("Multiindex has no free indices (it is not a tensor expression).")

    def is_cellwise_constant(self):
        """Check if cellwise constant.

        Always True.
        """
        return True

    def ufl_domains(self):
        """Return tuple of domains related to this terminal object."""
        return ()

    # --- Adding multiindices ---

    def __add__(self, other):
        """Add."""
        if isinstance(other, tuple):
            return MultiIndex(self._indices + other)
        elif isinstance(other, MultiIndex):
            return MultiIndex(self._indices + other._indices)
        return NotImplemented

    def __radd__(self, other):
        """Add."""
        if isinstance(other, tuple):
            return MultiIndex(other + self._indices)
        elif isinstance(other, MultiIndex):
            return MultiIndex(other._indices + self._indices)
        return NotImplemented

    # --- String formatting ---

    def __str__(self):
        """Format as a string."""
        return ", ".join(str(i) for i in self._indices)

    def __repr__(self):
        """Return representation."""
        return f"MultiIndex({self._indices!r})"

    # --- Iteration protocol ---
    def __len__(self):
        """Get length."""
        return len(self._indices)

    def __getitem__(self, i):
        """Get an item."""
        return self._indices[i]

    def __iter__(self):
        """Return iteratable."""
        return iter(self._indices)


def indices(n):
    """Return a tuple of n new Index objects."""
    return tuple(Index() for i in range(n))
