"""Base class for multifunctions with UFL ``Expr`` type dispatch."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

from inspect import signature

from ufl.core.ufl_type import registry, ufl_handler_names


def get_num_args(function):
    """Return the number of arguments accepted by *function*."""
    sig = signature(function)
    return len(sig.parameters) + 1


def memoized_handler(handler):
    """Function decorator to memoize ``MultiFunction`` handlers."""

    def _memoized_handler(self, o):
        c = getattr(self, "_memoized_handler_cache")
        r = c.get(o)
        if r is None:
            r = handler(self, o)
            c[o] = r
        return r

    return _memoized_handler


class MultiFunction:
    """Base class for collections of non-recursive expression node handlers.

    Subclass this (remember to call the ``__init__`` method of this class),
    and implement handler functions for each ``Expr`` type, using the lower case
    handler name of the type (``exprtype._ufl_handler_name_``).

    This class is optimized for efficient type based dispatch in the
    ``__call__``
    operator via typecode based lookup of the handler function bound to the
    algorithm object. Of course Python's function call overhead still applies.
    """

    _handlers_cache: dict[type, tuple[list[str], bool]] = {}

    def __init__(self):
        """Initialise."""
        self._update_handlers()

        # Create cache for memoized_handler
        self._memoized_handler_cache = {}

    def _update_handlers(self):
        """Build the handler table, indexed by typecode, for all registered types."""
        # The handler names are cached for each algorithm class, and rebuilt
        # when types have been registered since
        algorithm_class = type(self)
        cache_data = MultiFunction._handlers_cache.get(algorithm_class)
        if cache_data is None or len(cache_data[0]) != len(registry):
            handler_names = [None] * len(registry)

            # Use the handler of the closest registered type in the MRO
            for classobject in registry.classes:
                for handler_name in ufl_handler_names(classobject):
                    if hasattr(self, handler_name):
                        handler_names[classobject._ufl_typecode_] = handler_name
                        break
            is_cutoff_type = [get_num_args(getattr(self, name)) == 2 for name in handler_names]
            cache_data = (handler_names, is_cutoff_type)
            MultiFunction._handlers_cache[algorithm_class] = cache_data

        # Build handler list for this particular class (get functions
        # bound to self, these cannot be cached)
        handler_names, is_cutoff_type = cache_data
        self._handlers = [getattr(self, name) for name in handler_names]
        self._is_cutoff_type = is_cutoff_type

    def __call__(self, o, *args):
        """Delegate to handler function based on typecode of first argument."""
        try:
            handler = self._handlers[o._ufl_typecode_]
        except IndexError:
            # A type registered after this table was built
            self._update_handlers()
            handler = self._handlers[o._ufl_typecode_]
        return handler(o, *args)

    def undefined(self, o, *args):
        """Trigger error for types with missing handlers."""
        raise ValueError(f"No handler defined for {type(o).__name__}.")

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
