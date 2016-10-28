# -*- coding: utf-8 -*-

# Copyright (C) 2016-2016 Martin Sandve Alnæs
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

"""Python 2/3 compatibility utilities."""

import six


if not six.PY2:
    unicode = str

    def as_native_str(s):
        "Return s as unicode string, decoded using utf-8 if necessary."
        if isinstance(s, bytes):
            return s.decode("utf-8")
        else:
            return s
else:
    def as_native_str(s):
        "Return s as bytes string, encoded using utf-8 if necessary."
        if isinstance(s, unicode):
            return s.encode("utf-8")
        else:
            return s


def as_native_strings(stringlist):
    return [as_native_str(s) for s in stringlist]


def as_bytes(s):
    "Return s if bytes, or encode unicode string to bytes using utf-8."
    if isinstance(s, unicode):
        return s.encode("utf-8")
    else:
        return s


def as_unicode(s):
    "Return s if unicode string, or decode bytes to unicode string using utf-8."
    if isinstance(s, bytes):
        return s.decode("utf-8")
    else:
        return s
