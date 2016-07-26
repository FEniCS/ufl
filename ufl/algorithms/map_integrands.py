# -*- coding: utf-8 -*-
"""Basic algorithms for applying functions to subexpressions."""

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

# NOTE: Placing this under algorithms/ because I want corealg/ to stay clean
# as part of a careful refactoring process, and this file depends on ufl.form
# which drags in a lot of stuff.

from ufl.log import error
from ufl.core.expr import Expr
from ufl.corealg.map_dag import map_expr_dag
from ufl.integral import Integral
from ufl.form import Form
from ufl.constantvalue import Zero


def map_integrands(function, form, only_integral_type=None):
    """Apply transform(expression) to each integrand
    expression in form, or to form if it is an Expr.
    """

    if isinstance(form, Form):
        mapped_integrals = [map_integrands(function, itg, only_integral_type) for itg in form.integrals()]
        nonzero_integrals = [itg for itg in mapped_integrals if not isinstance(itg.integrand(), Zero)]
        return Form(nonzero_integrals)

    elif isinstance(form, Integral):
        itg = form
        if (only_integral_type is None) or (itg.integral_type() in only_integral_type):
            return itg.reconstruct(function(itg.integrand()))
        else:
            return itg

    elif isinstance(form, Expr):
        # ufl_assert(only_integral_type is None, "Restricting integral type is only valid with Form and Integral.")
        integrand = form
        return function(integrand)

    else:
        error("Expecting Form, Integral or Expr.")


def map_integrand_dags(function, form, only_integral_type=None, compress=True):
    return map_integrands(lambda expr: map_expr_dag(function, expr, compress),
                          form, only_integral_type)
