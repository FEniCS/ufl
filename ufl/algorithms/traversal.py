"""This module contains algorithms for traversing expression trees in different ways."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008

from ufl.action import Action
from ufl.adjoint import Adjoint
from ufl.core.expr import Expr
from ufl.form import BaseForm, Form, FormSum
from ufl.integral import Integral


def iter_expressions(a):
    """Utility function to handle Form, Integral and any Expr the same way when inspecting expressions.

    Returns an iterable over Expr instances:
    - a is an Expr: (a,)
    - a is an Integral:  the integrand expression of a
    - a is a  Form:      all integrand expressions of all integrals
    - a is a  FormSum:   the components of a
    - a is an Action:    the left and right component of a
    - a is an Adjoint:   the underlying form of a
    """
    if isinstance(a, Form):
        return (itg.integrand() for itg in a.integrals())
    elif isinstance(a, Integral):
        return (a.integrand(),)
    elif isinstance(a, (FormSum, Adjoint, Action)):
        return tuple(e for op in a.ufl_operands for e in iter_expressions(op))
    elif isinstance(a, (Expr, BaseForm)):
        return (a,)
    raise ValueError(f"Not an UFL type: {type(a)}")
