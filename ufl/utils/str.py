# -*- coding: utf-8 -*-

# Copyright (C) 2016-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""String manipulation utilities."""


def as_native_str(s):
    "Return s as unicode string, decoded using utf-8 if necessary."
    if isinstance(s, bytes):
        return s.decode("utf-8")
    else:
        return s


def as_native_strings(stringlist):
    return [as_native_str(s) for s in stringlist]
