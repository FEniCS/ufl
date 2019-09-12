# -*- coding: utf-8 -*-
"Utilites for types with a global unique counter attached to each object."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


def counted_init(self, count=None, countedclass=None):
    "Initialize a counted object, see ExampleCounted below for how to use."

    if countedclass is None:
        countedclass = type(self)

    if count is None:
        count = countedclass._globalcount

    self._count = count

    if self._count >= countedclass._globalcount:
        countedclass._globalcount = self._count + 1


class ExampleCounted(object):
    """An example class for classes of objects identified by a global counter.

    Mimic this class to create globally counted objects within a single type.
    """
    # Store the count for each object
    __slots__ = ("_count",)

    # Store a global counter with the class
    _globalcount = 0

    # Call counted_init with an optional constructor argument and the class
    def __init__(self, count=None):
        counted_init(self, count, ExampleCounted)

    # Make the count accessible
    def count(self):
        return self._count
