"""This module defines the Terminal class.

Terminal the superclass for all types that are terminal nodes in an expression tree.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008
# Modified by Massimiliano Leoni, 2016

import warnings

from ufl.core.expr import Expr
from ufl.core.ufl_type import ufl_type


@ufl_type(is_abstract=True, is_terminal=True)
class Terminal(Expr):
    """Base class for terminal objects.

    A terminal node in the UFL expression tree.
    """

    __slots__ = ()

    def __init__(self):
        """Initialise the terminal."""
        Expr.__init__(self)

    ufl_operands = ()
    ufl_free_indices = ()
    ufl_index_dimensions = ()

    def ufl_domains(self):
        """Return tuple of domains related to this terminal object."""
        raise NotImplementedError("Missing implementation of domains().")

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        """Get *self* from *mapping* and return the component asked for."""
        f = mapping.get(self)
        # No mapping, trying to evaluate self as a constant
        if f is None:
            try:
                try:
                    f = float(self)
                except TypeError:
                    f = complex(self)
                if derivatives:
                    f = 0.0
                return f
            except Exception:
                pass
            # If it has an ufl_evaluate function, call it
            if hasattr(self, 'ufl_evaluate'):
                return self.ufl_evaluate(x, component, derivatives)
            # Take component if any
            warnings.warn("Couldn't map '%s' to a float, returning ufl object without evaluation." % str(self))
            f = self
            if component:
                f = f[component]
            return f

        # Found a callable in the mapping
        if callable(f):
            if derivatives:
                f = f(x, derivatives)
            else:
                f = f(x)
        else:
            if derivatives:
                return 0.0

        # Take component if any (expecting nested tuple)
        for c in component:
            f = f[c]
        return f

    def _ufl_signature_data_(self, renumbering):
        """Default signature data for of terminals just return the repr string."""
        return repr(self)

    def _ufl_compute_hash_(self):
        """Default hash of terminals just hash the repr string."""
        return hash(repr(self))

    def __eq__(self, other):
        """Default comparison of terminals just compare repr strings."""
        return repr(self) == repr(other)


# --- Subgroups of terminals ---

@ufl_type(is_abstract=True)
class FormArgument(Terminal):
    """An abstract class for a form argument (a thing in a primal finite element space)."""
    __slots__ = ()

    def __init__(self):
        """Initialise the form argument."""
        Terminal.__init__(self)
