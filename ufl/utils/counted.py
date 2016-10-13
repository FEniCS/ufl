# -*- coding: utf-8 -*-
"Utilites for types with a global unique counter attached to each object."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
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


from ufl.utils.py23 import as_native_strings


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
    __slots__ = as_native_strings(("_count",))

    # Store a global counter with the class
    _globalcount = 0

    # Call counted_init with an optional constructor argument and the class
    def __init__(self, count=None):
        counted_init(self, count, ExampleCounted)

    # Make the count accessible
    def count(self):
        return self._count
