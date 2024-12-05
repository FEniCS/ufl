"""Mixin class for types with a global unique counter attached to each object."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import itertools


class Counted:
    """Mixin class for globally counted objects."""

    # Mixin classes do not work well with __slots__ so _count must be
    # added to the __slots__ of the inheriting class
    __slots__ = ()

    _counter = None

    def __init__(self, count=None, counted_class=None):
        """Initialize the Counted instance.

        Args:
            count: The object count, if ``None`` defaults to the next value
                according to the global counter (per type).
            counted_class: Class to attach the global counter too. If ``None``
                then ``type(self)`` will be used.

        """
        # create a new counter for each subclass
        counted_class = counted_class or type(self)
        if counted_class._counter is None:
            counted_class._counter = itertools.count()

        self._count = count if count is not None else next(counted_class._counter)
        self._counted_class = counted_class

    def count(self):
        """Get count."""
        return self._count
