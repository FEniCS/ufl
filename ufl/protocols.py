# -*- coding: utf-8 -*-
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


def id_or_none(obj):
    """Returns None if the object is None, obj.ufl_id() if available, or id(obj) if not.

    This allows external libraries to implement an alternative
    to id(obj) in the ufl_id() function, such that ufl can identify
    objects as the same without knowing about their types.
    """
    if obj is None:
        return None
    elif hasattr(obj, 'ufl_id'):
        return obj.ufl_id()
    else:
        return id(obj)


def metadata_equal(a, b):
    return sorted((k, id(v)) for k, v in list(a.items())) == sorted((k, id(v)) for k, v in list(b.items()))


def metadata_hashdata(md):
    return tuple(sorted((k, id(v)) for k, v in list(md.items())))
