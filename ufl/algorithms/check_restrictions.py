"""Algorithms related to restrictions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

from functools import singledispatchmethod

from ufl.classes import Expr, FacetNormal, FormArgument, Restricted
from ufl.corealg.dag_traverser import DAGTraverser


class RestrictionChecker(DAGTraverser):
    """Restiction checker."""

    def __init__(self, require_restriction):
        """Initialise."""
        super().__init__()
        self.current_restriction = None
        self.require_restriction = require_restriction

    @singledispatchmethod
    def process(self, o: Expr):
        """Process an expression."""
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr):
        """Check only explicitly handled nodes.

        This is intentionally a cutoff handler, matching the old
        ``MultiFunction`` implementation.  Restricted nodes recurse below
        their operand explicitly so that the state is set only for that
        subtree.
        """
        pass

    @process.register(Restricted)
    def _(self, o: Restricted):
        """Apply to restricted."""
        if self.current_restriction is not None:
            raise ValueError("Not expecting twice restricted expression.")
        self.current_restriction = o._side
        (e,) = o.ufl_operands
        try:
            self(e)
        finally:
            self.current_restriction = None

    @process.register(FacetNormal)
    def _(self, o: FacetNormal):
        """Apply to facet_normal."""
        if self.require_restriction:
            if self.current_restriction is None:
                raise ValueError("Facet normal must be restricted in interior facet integrals.")
        else:
            if self.current_restriction is not None:
                raise ValueError("Restrictions are only allowed for interior facet integrals.")

    @process.register(FormArgument)
    def _(self, o: FormArgument):
        """Apply to form_argument."""
        if self.require_restriction:
            if self.current_restriction is None:
                raise ValueError("Form argument must be restricted in interior facet integrals.")
        else:
            if self.current_restriction is not None:
                raise ValueError("Restrictions are only allowed for interior facet integrals.")


def check_restrictions(expression, require_restriction):
    """Check that types that must be restricted are restricted in expression."""
    rules = RestrictionChecker(require_restriction)
    return rules(expression)
