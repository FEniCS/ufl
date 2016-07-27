# -*- coding: utf-8 -*-
"""This module defines the ``Terminal`` class, the superclass
for all types that are terminal nodes in an expression tree."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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
#
# Modified by Anders Logg, 2008
# Modified by Massimiliano Leoni, 2016

from ufl.log import error, warning
from ufl.core.expr import Expr
from ufl.core.ufl_type import ufl_type


# --- Base class for terminal objects ---

@ufl_type(is_abstract=True, is_terminal=True)
class Terminal(Expr):
    "A terminal node in the UFL expression tree."
    __slots__ = ()

    def __init__(self):
        Expr.__init__(self)

    def _ufl_expr_reconstruct_(self, *operands):
        "Return self."
        if operands:
            error("Terminal has no operands.")
        return self

    ufl_operands = ()
    ufl_free_indices = ()
    ufl_index_dimensions = ()

    def ufl_domains(self):
        "Return tuple of domains related to this terminal object."
        raise NotImplementedError("Missing implementation of domains().")

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get *self* from *mapping* and return the component asked for."
        f = mapping.get(self)
        # No mapping, trying to evaluate self as a constant
        if f is None:
            try:
                f = float(self)
                if derivatives:
                    f = 0.0
                return f
            except:
                pass
            # If it has an ufl_evaluate function, call it
            if hasattr(self, 'ufl_evaluate'):
                return self.ufl_evaluate(x, component, derivatives)
            # Take component if any
            warning("Couldn't map '%s' to a float, returning ufl object without evaluation." % str(self))
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
        "Default signature data for of terminals just return the repr string."
        return repr(self)

    def _ufl_compute_hash_(self):
        "Default hash of terminals just hash the repr string."
        return hash(repr(self))

    def __eq__(self, other):
        "Default comparison of terminals just compare repr strings."
        return repr(self) == repr(other)


# --- Subgroups of terminals ---

@ufl_type(is_abstract=True)
class FormArgument(Terminal):
    "An abstract class for a form argument."
    __slots__ = ()

    def __init__(self):
        Terminal.__init__(self)
