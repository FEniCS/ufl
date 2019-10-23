# -*- coding: utf-8 -*-
"""This module contains algorithms for traversing expression trees in different ways."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008

from ufl.log import error
from ufl.core.expr import Expr
from ufl.integral import Integral
from ufl.form import Form


# --- Traversal utilities ---

def iter_expressions(a):
    """Utility function to handle Form, Integral and any Expr
    the same way when inspecting expressions.
    Returns an iterable over Expr instances:
    - a is an Expr: (a,)
    - a is an Integral:  the integrand expression of a
    - a is a  Form:      all integrand expressions of all integrals
    """
    if isinstance(a, Form):
        return (itg.integrand() for itg in a.integrals())
    elif isinstance(a, Integral):
        return (a.integrand(),)
    elif isinstance(a, Expr):
        return (a,)
    error("Not an UFL type: %s" % str(type(a)))
