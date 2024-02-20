"""Base class for multifunctions with UFL ``Expr`` type dispatch."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

import inspect
import typing

from ufl.core.ufl_type import UFLObject


def get_num_args(function):
    """Return the number of arguments accepted by *function*."""
    sig = inspect.signature(function)
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


class MultiFunction(UFLObject):
    """Base class for collections of non-recursive expression node handlers.

    Subclass this (remember to call the ``__init__`` method of this class),
    and implement handler functions for each ``Expr`` type, using the lower case
    handler name of the type (``exprtype._ufl_handler_name_``).

    This class is optimized for efficient type based dispatch in the
    ``__call__``
    operator via typecode based lookup of the handler function bound to the
    algorithm object. Of course Python's function call overhead still applies.
    """

    _handlers_cache: typing.Dict[type, typing.Tuple[typing.List[str], bool]] = {}

    def __init__(self):
        """Initialise."""
        # Analyse class properties and cache handler data the
        # first time this is run for a particular class
        # (cached for each algorithm for performance)
        algorithm_class = type(self)
        cache_data = MultiFunction._handlers_cache.get(algorithm_class)
        if not cache_data:
            cache_data = ({}, {})
            MultiFunction._handlers_cache[algorithm_class] = cache_data

        # Build handler list for this particular class (get functions
        # bound to self, these cannot be cached)
        handler_names, self._is_cutoff_type = cache_data
        self._handlers = {o: getattr(self, name) for o, name in handler_names.items()}

        # Create cache for memoized_handler
        self._memoized_handler_cache = {}

    def __call__(self, o, *args):
        """Delegate to handler function based on typecode of first argument."""
        t = o.__class__
        if t not in self._handlers:
            for c in t.mro():
                try:
                    hname = c._ufl_handler_name_
                except AttributeError:
                    hname = UFLObject._ufl_handler_name_
                if hasattr(self, hname):
                    self._handlers[t] = getattr(self, hname)
                    self._is_cutoff_type[t] = get_num_args(self._handlers[t]) == 2
                    break
            else:
                self._handlers[t] = self.undefined
                self._is_cutoff_type[t] = False
        return self._handlers[t](o, *args)

    def undefined(self, o, *args):
        """Trigger error for types with missing handlers."""
        raise ValueError(f"No handler defined for {o.__class__.__name__}.")

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

    def __repr__(self):
        """Representation."""
        raise NotImplementedError()

    def __str__(self):
        """String representation."""
        raise NotImplementedError()

    def _ufl_hash_data_(self):
        raise NotImplementedError()

    # Set default behaviour for any UFLObject as undefined
    ufl_type = undefined
