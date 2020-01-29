# -*- coding: utf-8 -*-
"""Basic algorithms for applying functions to subexpressions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

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
        mapped_integrals = [map_integrands(function, itg, only_integral_type)
                            for itg in form.integrals()]
        nonzero_integrals = [itg for itg in mapped_integrals
                             if not isinstance(itg.integrand(), Zero)]
        return Form(nonzero_integrals)
    elif isinstance(form, Integral):
        itg = form
        if (only_integral_type is None) or (itg.integral_type() in only_integral_type):
            return itg.reconstruct(function(itg.integrand()))
        else:
            return itg
    elif isinstance(form, Expr):
        integrand = form
        return function(integrand)
    else:
        error("Expecting Form, Integral or Expr.")


def map_integrand_dags(function, form, only_integral_type=None, compress=True):
    return map_integrands(lambda expr: map_expr_dag(function, expr, compress),
                          form, only_integral_type)
