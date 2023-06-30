# -*- coding: utf-8 -*-
"Mixin class for types with a global unique counter attached to each object."

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

    def __init__(self, count=None, cls=None):
        """Initialize the Counted instance.

        :arg count: The object count, if ``None`` defaults to the next value
                    according to the global counter (per type).
        :arg cls: Class to attach the global counter too. If ``None`` then
                  ``type(self)`` will be used.

        """
        # create a new counter for each subclass
        cls = cls or type(self)
        if cls._counter is None:
            cls._counter = itertools.count()

        self._count = count if count is not None else next(cls._counter)

    def count(self):
        return self._count
