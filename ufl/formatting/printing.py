# -*- coding: utf-8 -*-
"""A collection of utility algorithms for printing
of UFL objects, mostly intended for debugging purposes."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg 2009, 2014

from ufl.log import error
from ufl.core.expr import Expr
from ufl.form import Form
from ufl.integral import Integral


# --- Utilities for constructing informative strings from UFL objects

def integral_info(integral):
    if not isinstance(integral, Integral):
        error("Expecting an Integral.")
    s = "  Integral:\n"
    s += "    Type:\n"
    s += "      %s\n" % integral.integral_type()
    s += "    Domain:\n"
    s += "      %s\n" % integral.ufl_domain()
    s += "    Domain id:\n"
    s += "      %s\n" % integral.subdomain_id()
    s += "    Domain data:\n"
    s += "      %s\n" % integral.subdomain_data()
    s += "    Compiler metadata:\n"
    s += "      %s\n" % integral.metadata()
    return s


def form_info(form):
    if not isinstance(form, Form):
        error("Expecting a Form.")

    bf = form.arguments()
    cf = form.coefficients()

    s = "Form info:\n"
    s += "  rank:                          %d\n" % len(bf)
    s += "  num_coefficients:              %d\n" % len(cf)
    s += "\n"

    for f in cf:
        if f._name:
            s += "\n"
            s += "  Coefficient %d is named '%s'" % (f._count, f._name)
    s += "\n"

    integrals = form.integrals()
    integral_types = sorted(set(itg.integral_type() for itg in integrals))
    for integral_type in integral_types:
        itgs = form.integrals_by_type(integral_type)
        s += "  num_{0}_integrals:  {1}\n".format(integral_type, len(itgs))
    s += "\n"

    for integral_type in integral_types:
        itgs = form.integrals_by_type(integral_type)
        for itg in itgs:
            s += integral_info(itg)
            s += "\n"

    return s


def _indent_string(n):
    return "    " * n


def _tree_format_expression(expression, indentation, parentheses):
    ind = _indent_string(indentation)
    if expression._ufl_is_terminal_:
        s = "%s%s" % (ind, repr(expression))
    else:
        sops = [_tree_format_expression(o, indentation + 1, parentheses) for o in expression.ufl_operands]
        s = "%s%s\n" % (ind, expression._ufl_class_.__name__)
        if parentheses and len(sops) > 1:
            s += "%s(\n" % (ind,)
        s += "\n".join(sops)
        if parentheses and len(sops) > 1:
            s += "\n%s)" % (ind,)
    return s


def tree_format(expression, indentation=0, parentheses=True):
    s = ""

    if isinstance(expression, Form):
        form = expression
        integrals = form.integrals()
        integral_types = sorted(set(itg.integral_type() for itg in integrals))
        itgs = []
        for integral_type in integral_types:
            itgs += list(form.integrals_by_type(integral_type))

        ind = _indent_string(indentation)
        s += ind + "Form:\n"
        s += "\n".join(tree_format(itg, indentation + 1, parentheses) for itg in itgs)

    elif isinstance(expression, Integral):
        ind = _indent_string(indentation)
        s += ind + "Integral:\n"
        ind = _indent_string(indentation + 1)
        s += ind + "integral type: %s\n" % expression.integral_type()
        s += ind + "subdomain id: %s\n" % expression.subdomain_id()
        s += ind + "integrand:\n"
        s += tree_format(expression._integrand, indentation + 2, parentheses)

    elif isinstance(expression, Expr):
        s += _tree_format_expression(expression, indentation, parentheses)

    else:
        error("Invalid object type %s" % type(expression))

    return s
