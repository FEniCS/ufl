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

import numpy as np
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
            if hasattr(self, "ufl_evaluate"):
                return self.ufl_evaluate(x, component, derivatives)
            # Take component if any
            warnings.warn(
                f"Couldn't map '{self}' to a float, returning ufl object without evaluation."
            )
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

    def traverse_dag_apply_coefficient_split(
        self,
        coefficient_split,
        reference_value=False,
        reference_grad=0,
        restricted=None,
        cache=None,
    ):
        from ufl.classes import (
            ComponentTensor,
            MultiIndex,
            NegativeRestricted,
            PositiveRestricted,
            ReferenceGrad,
            ReferenceValue,
            Zero,
        )
        from ufl.core.multiindex import indices
        from ufl.checks import is_cellwise_constant
        from ufl.domain import extract_unique_domain
        from ufl.tensors import as_tensor

        c = self
        if reference_value:
            c = ReferenceValue(c)
        for _ in range(reference_grad):
            # Return zero if expression is trivially constant. This has to
            # happen here because ReferenceGrad has no access to the
            # topological dimension of a literal zero.
            if is_cellwise_constant(c):
                dim = extract_unique_domain(subcoeff).topological_dimension()
                c = Zero(c.ufl_shape + (dim,), c.ufl_free_indices, c.ufl_index_dimensions)
            else:
                c = ReferenceGrad(c)
        if restricted == "+":
            c = PositiveRestricted(c)
        elif restricted == "-":
            c = NegativeRestricted(c)
        elif restricted is not None:
            raise RuntimeError(f"Got unknown restriction: {restricted}")
        return c

# --- Subgroups of terminals ---


@ufl_type(is_abstract=True)
class FormArgument(Terminal):
    """An abstract class for a form argument (a thing in a primal finite element space)."""

    __slots__ = ()

    def __init__(self):
        """Initialise the form argument."""
        Terminal.__init__(self)
