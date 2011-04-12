"""A collection of utility algorithms for printing
of UFL objects, mostly intended for debugging purposes."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-03-14 -- 2008-10-21"

# Modified by Anders Logg, 2009.

from itertools import chain

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.form import Form
from ufl.integral import Integral
from ufl.algorithms.analysis import extract_arguments, extract_coefficients

#--- Utilities for constructing informative strings from UFL objects ---

def integral_info(integral):
    ufl_assert(isinstance(integral, Integral), "Expecting an Integral.")
    s  = "  Integral:\n"
    s += "    Measure representation:\n"
    s += "      %r\n" % integral.measure()
    s += "    Integrand expression representation:\n"
    s += "      %r\n" % integral.integrand()
    s += "    Integrand expression short form:\n"
    s += "      %s" % integral.integrand()
    return s

def form_info(form):
    ufl_assert(isinstance(form, Form), "Expecting a Form.")

    bf = extract_arguments(form)
    cf = extract_coefficients(form)

    ci = form.cell_integrals()
    ei = form.exterior_facet_integrals()
    ii = form.interior_facet_integrals()
    mi = form.macro_cell_integrals()

    s  = "Form info:\n"
    s += "  rank:                          %d\n" % len(bf)
    s += "  num_coefficients:              %d\n" % len(cf)
    s += "  num_cell_integrals:            %d\n" % len(ci)
    s += "  num_exterior_facet_integrals:  %d\n" % len(ei)
    s += "  num_interior_facet_integrals:  %d\n" % len(ii)
    s += "  num_macro_cell_integrals:      %d\n" % len(mi)

    for f in cf:
        if f._name:
            s += "\n"
            s += "  Function %d is named '%s'" % (f._count, f._name)
    s += "\n"

    for itg in ci:
        s += "\n"
        s += integral_info(itg)
    for itg in ei:
        s += "\n"
        s += integral_info(itg)
    for itg in ii:
        s += "\n"
        s += integral_info(itg)
    for itg in mi:
        s += "\n"
        s += integral_info(itg)
    return s

def _indent_string(n):
    return "    "*n

def _tree_format_expression(expression, indentation, parentheses):
    ind = _indent_string(indentation)
    if isinstance(expression, Terminal):
        s = ind + "%s" % repr(expression)
    else:
        sops = [_tree_format_expression(o, indentation+1, parentheses) for o in expression.operands()]
        s = ind + "%s\n" % expression._uflclass.__name__
        if parentheses and len(sops) > 1:
            s += ind + "(\n"
        s += "\n".join(sops)
        if parentheses and len(sops) > 1:
            s += "\n" + ind + ")"
    return s

def tree_format(expression, indentation=0, parentheses=True):
    s = ""

    if isinstance(expression, Form):
        ci = expression.cell_integrals()
        ei = expression.exterior_facet_integrals()
        ii = expression.interior_facet_integrals()
        mi = expression.macro_cell_integrals()
        ind = _indent_string(indentation)
        s += ind + "Form:\n"
        s += "\n".join(tree_format(itg, indentation+1, parentheses) for itg in chain(ci, ei, ii, mi))

    elif isinstance(expression, Integral):
        ind = _indent_string(indentation)
        s += ind + "Integral:\n"
        ind = _indent_string(indentation+1)
        s += ind + "domain type: %s\n" % expression.measure().domain_type()
        s += ind + "domain id: %d\n" % expression.measure().domain_id()
        s += ind + "integrand:\n"
        s += tree_format(expression._integrand, indentation+2, parentheses)

    elif isinstance(expression, Expr):
        s += _tree_format_expression(expression, indentation, parentheses)

    else:
        error("Invalid object type %s" % type(expression))

    return s

