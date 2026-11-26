"""Transformer.

This module defines the Transformer base class and some
basic specializations to further base other algorithms upon,
as well as some utilities for easier application of such
algorithms.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

import inspect

from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import Variable
from ufl.core.ufl_type import registry, ufl_handler_names


def is_post_handler(function):
    """Check if function is a handler that expects transformed children as input."""
    insp = inspect.getfullargspec(function)
    num_args = len(insp[0]) + int(insp[1] is not None)
    visit_children_first = num_args > 2
    return visit_children_first


class Transformer:
    """Transformer.

    Base class for a visitor-like algorithm design pattern used to
    transform expression trees from one representation to another.
    """

    _handlers_cache: dict[type, tuple[str, bool]] = {}

    def __init__(self, variable_cache=None):
        """Initialise."""
        if variable_cache is None:
            variable_cache = {}
        self._variable_cache = variable_cache
        self._update_handlers()
        # Keep a stack of objects visit is called on, to ease
        # backtracking
        self._visit_stack = []

    def _update_handlers(self):
        """Build the handler table, indexed by typecode, for all registered types."""
        # The handler data is cached for each transformer class, and rebuilt
        # when types have been registered since
        cache_data = Transformer._handlers_cache.get(type(self))
        if cache_data is None or len(cache_data) != len(registry):
            cache_data = [None] * len(registry)
            # Use the handler of the closest registered type in the MRO
            for classobject in registry.classes:
                for handler_name in ufl_handler_names(classobject):
                    function = getattr(self, handler_name, None)
                    if function:
                        cache_data[classobject._ufl_typecode_] = (
                            handler_name,
                            is_post_handler(function),
                        )
                        break
            Transformer._handlers_cache[type(self)] = cache_data

        # Build handler list for this particular class (get functions
        # bound to self)
        self._handlers = [(getattr(self, name), post) for (name, post) in cache_data]

    def print_visit_stack(self):
        """Print visit stack."""
        print("/" * 80)
        print("Visit stack in Transformer:")

        def sstr(s):
            """Format."""
            ss = str(type(s)) + " ; "
            n = 160 - len(ss)
            return ss + str(s)[:n]

        print("\n".join(map(sstr, self._visit_stack)))
        print("\\" * 80)

    def visit(self, o):
        """Visit."""
        # Update stack
        self._visit_stack.append(o)

        # Get handler for the UFL class of o (type(o) may be an
        # external subclass of the actual UFL class)
        try:
            h, visit_children_first = self._handlers[o._ufl_typecode_]
        except IndexError:
            # A type registered after this table was built
            self._update_handlers()
            h, visit_children_first = self._handlers[o._ufl_typecode_]

        # Is this a handler that expects transformed children as
        # input?
        if visit_children_first:
            # Yes, visit all children first and then call h.
            r = h(o, *map(self.visit, o.ufl_operands))
        else:
            # No, this is a handler that handles its own children
            # (arguments self and o, where self is already bound)
            r = h(o)

        # Update stack and return
        self._visit_stack.pop()
        return r

    def undefined(self, o):
        """Trigger error."""
        raise ValueError(f"No handler defined for {type(o).__name__}.")

    def reuse(self, o):
        """Reuse Expr (ignore children)."""
        return o

    def reuse_if_untouched(self, o, *ops):
        """Reuse object if operands are the same objects.

        Use in your own subclass by setting e.g. `expr = MultiFunction.reuse_if_untouched`
        as a default rule.
        """
        if all(a is b for a, b in zip(o.ufl_operands, ops)):
            return o
        else:
            return o._ufl_expr_reconstruct_(*ops)

    # It's just so slow to compare all operands, avoiding it now
    reuse_if_possible = reuse_if_untouched

    def always_reconstruct(self, o, *operands):
        """Reconstruct expr."""
        return o._ufl_expr_reconstruct_(*operands)

    # Set default behaviour for any UFLType
    ufl_type = undefined

    # Set default behaviour for any Terminal
    terminal = reuse

    def reuse_variable(self, o):
        """Reuse variable."""
        # Check variable cache to reuse previously transformed
        # variable if possible
        e, l = o.ufl_operands  # noqa: E741
        v = self._variable_cache.get(l)
        if v is not None:
            return v

        # Visit the expression our variable represents
        e2 = self.visit(e)

        # If the expression is the same, reuse Variable object
        if e == e2:
            v = o
        else:
            # Recreate Variable (with same label)
            v = Variable(e2, l)

        # Cache variable
        self._variable_cache[l] = v
        return v

    def reconstruct_variable(self, o):
        """Reconstruct variable."""
        # Check variable cache to reuse previously transformed
        # variable if possible
        e, l = o.ufl_operands  # noqa: E741
        v = self._variable_cache.get(l)
        if v is not None:
            return v

        # Visit the expression our variable represents
        e2 = self.visit(e)

        # Always reconstruct Variable (with same label)
        v = Variable(e2, l)
        self._variable_cache[l] = v
        return v


class ReuseTransformer(Transformer):
    """Reuse transformer."""

    def __init__(self, variable_cache=None):
        """Initialise."""
        Transformer.__init__(self, variable_cache)

    # Set default behaviour for any Expr
    expr = Transformer.reuse_if_untouched

    # Set default behaviour for any Terminal
    terminal = Transformer.reuse

    # Set default behaviour for Variable
    variable = Transformer.reuse_variable


class CopyTransformer(Transformer):
    """Copy transformer."""

    def __init__(self, variable_cache=None):
        """Initialise."""
        Transformer.__init__(self, variable_cache)

    # Set default behaviour for any Expr
    expr = Transformer.always_reconstruct

    # Set default behaviour for any Terminal
    terminal = Transformer.reuse

    # Set default behaviour for Variable
    variable = Transformer.reconstruct_variable


class VariableStripper(ReuseTransformer):
    """Variable stripper."""

    def __init__(self):
        """Initialise."""
        ReuseTransformer.__init__(self)

    def variable(self, o):
        """Visit a variable."""
        return self.visit(o.ufl_operands[0])


def apply_transformer(e, transformer, integral_type=None):
    """Apply transforms.

    Apply transformer.visit(expression) to each integrand expression in
    form, or to form if it is an Expr.
    """
    return map_integrands(transformer.visit, e, integral_type)


def strip_variables(e):
    """Replace all Variable instances with the expression they represent."""
    return apply_transformer(e, VariableStripper())
