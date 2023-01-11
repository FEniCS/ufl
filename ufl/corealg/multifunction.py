# -*- coding: utf-8 -*-
"""Base class for multifunctions with UFL ``Expr`` type dispatch."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

import inspect

from ufl.core.expr import Expr
from ufl.core.ufl_type import UFLType


def get_num_args(function):
    "Return the number of arguments accepted by *function*."
    sig = inspect.signature(function)
    return len(sig.parameters) + 1


def memoized_handler(handler):
    "Function decorator to memoize ``MultiFunction`` handlers."

    def _memoized_handler(self, o):
        c = getattr(self, "_memoized_handler_cache")
        r = c.get(o)
        if r is None:
            r = handler(self, o)
            c[o] = r
        return r
    return _memoized_handler


class MultiFunction(object):
    """Base class for collections of non-recursive expression node handlers.

    Subclass this (remember to call the ``__init__`` method of this class),
    and implement handler functions for each ``Expr`` type, using the lower case
    handler name of the type (``exprtype._ufl_handler_name_``).

    This class is optimized for efficient type based dispatch in the
    ``__call__``
    operator via typecode based lookup of the handler function bound to the
    algorithm object. Of course Python's function call overhead still applies.
    """

    _handlers_cache = {}

    def __init__(self):
        # Analyse class properties and cache handler data the
        # first time this is run for a particular class
        # (cached for each algorithm for performance)
        algorithm_class = type(self)
        cache_data = MultiFunction._handlers_cache.get(algorithm_class)
        if not cache_data:
            handler_names = [None] * len(Expr._ufl_all_classes_)

            # Iterate over the inheritance chain for each Expr
            # subclass (NB! This assumes that all UFL classes inherits
            # from a single Expr subclass and that the first
            # superclass is always from the UFL Expr hierarchy!)
            for classobject in Expr._ufl_all_classes_:
                for c in classobject.mro():
                    # Register classobject with handler for the first
                    # encountered superclass
                    try:
                        handler_name = c._ufl_handler_name_
                    except AttributeError as attribute_error:
                        if type(classobject) is not UFLType:
                            raise attribute_error
                        # Default handler name for UFL types
                        handler_name = UFLType._ufl_handler_name_

                    if hasattr(self, handler_name):
                        handler_names[classobject._ufl_typecode_] = handler_name
                        break
            is_cutoff_type = [get_num_args(getattr(self, name)) == 2
                              for name in handler_names]
            cache_data = (handler_names, is_cutoff_type)
            MultiFunction._handlers_cache[algorithm_class] = cache_data

        # Build handler list for this particular class (get functions
        # bound to self, these cannot be cached)
        handler_names, is_cutoff_type = cache_data
        self._handlers = [getattr(self, name) for name in handler_names]
        self._is_cutoff_type = is_cutoff_type

        # Create cache for memoized_handler
        self._memoized_handler_cache = {}

    def __call__(self, o, *args):
        "Delegate to handler function based on typecode of first argument."
        return self._handlers[o._ufl_typecode_](o, *args)

    def undefined(self, o, *args):
        "Trigger error for types with missing handlers."
        raise ValueError(f"No handler defined for {o._ufl_class_.__name__}.")

    def reuse_if_untouched(self, o, *ops):
        """Reuse object if operands are the same objects.

        Use in your own subclass by setting e.g.
        ::

            expr = MultiFunction.reuse_if_untouched

        as a default rule.
        """
        if all(a is b for a, b in zip(o.ufl_operands, ops)):
            return o
        else:
            return o._ufl_expr_reconstruct_(*ops)

    # Set default behaviour for any UFLType as undefined
    ufl_type = undefined
