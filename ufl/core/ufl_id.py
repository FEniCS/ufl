# -*- coding: utf-8 -*-
"Utilites for types with a globally counted unique id attached to each object."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

from ufl.utils.str import as_native_str


def attach_ufl_id(cls):
    """Equip class with ``.ufl_id()`` and handle bookkeeping.

    Usage:

        1. Apply to class::

            @attach_ufl_id
            class MyClass(object):

        2. If ``__slots__`` is defined, include ``_ufl_id`` attribute::

            __slots__ = ("_ufl_id",)

        3. Add keyword argument to constructor::

            def __init__(self, *args, ufl_id=None):

        4. Call ``self._init_ufl_id`` with ``ufl_id`` and assign to ``._ufl_id``
           attribute::

            self._ufl_id = self._init_ufl_id(ufl_id)

    Result:

        ``MyClass().ufl_id()`` returns unique value for each constructed object.

    """

    def _get_ufl_id(self):
        "Return the ufl_id of this object."
        return self._ufl_id

    def _init_ufl_id(cls):
        "Initialize new ufl_id for the object under construction."
        # Bind cls with closure here

        def init_ufl_id(self, ufl_id):
            if ufl_id is None:
                ufl_id = cls._ufl_global_id
            cls._ufl_global_id = max(ufl_id, cls._ufl_global_id) + 1
            return ufl_id
        return init_ufl_id

    # Modify class:
    if hasattr(cls, "__slots__"):
        assert as_native_str("_ufl_id") in cls.__slots__
    cls._ufl_global_id = 0
    cls.ufl_id = _get_ufl_id
    cls._init_ufl_id = _init_ufl_id(cls)
    return cls
