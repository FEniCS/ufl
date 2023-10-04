"""Algorithms related to restrictions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag


class RestrictionChecker(MultiFunction):
    """Restiction checker."""

    def __init__(self, require_restriction):
        """Initialise."""
        MultiFunction.__init__(self)
        self.current_restriction = None
        self.require_restriction = require_restriction

    def expr(self, o):
        """Apply to expr."""
        pass

    def restricted(self, o):
        """Apply to restricted."""
        if self.current_restriction is not None:
            raise ValueError("Not expecting twice restricted expression.")
        self.current_restriction = o._side
        e, = o.ufl_operands
        self.visit(e)
        self.current_restriction = None

    def facet_normal(self, o):
        """Apply to facet_normal."""
        if self.require_restriction:
            if self.current_restriction is None:
                raise ValueError("Facet normal must be restricted in interior facet integrals.")
        else:
            if self.current_restriction is not None:
                raise ValueError("Restrictions are only allowed for interior facet integrals.")

    def form_argument(self, o):
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
    return map_expr_dag(rules, expression)
