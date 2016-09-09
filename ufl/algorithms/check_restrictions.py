# -*- coding: utf-8 -*-
"Algorithms related to restrictions."

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

from ufl.core.expr import Expr
from ufl.assertions import ufl_assert

from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag


class RestrictionChecker(MultiFunction):
    def __init__(self, require_restriction):
        MultiFunction.__init__(self)
        self.current_restriction = None
        self.require_restriction = require_restriction

    def expr(self, o):
        pass

    def restricted(self, o):
        ufl_assert(self.current_restriction is None,
                   "Not expecting twice restricted expression.")
        self.current_restriction = o._side
        e, = o.ufl_operands
        self.visit(e)
        self.current_restriction = None

    def facet_normal(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None,
                       "Facet normal must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None,
                       "Restrictions are only allowed for interior facet integrals.")

    def form_argument(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None,
                       "Form argument must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None,
                       "Restrictions are only allowed for interior facet integrals.")


def check_restrictions(expression, require_restriction):
    "Check that types that must be restricted are restricted in expression."
    rules = RestrictionChecker(require_restriction)
    return map_expr_dag(rules, expression)
