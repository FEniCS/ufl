"""A collection of utility algorithms for printing
of UFL objects, mostly intended for debugging purposers."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-21"

from itertools import chain

from ..output import ufl_assert
from ..base import UFLObject, Terminal
from ..form import Form
from ..integral import Integral

#--- Utilities for constructing informative strings from UFL objects ---

def integral_info(integral):
    s  = "  Integral over %s domain %d:\n" % (integral.domain_type(), integral.domain_id())
    s += "    Integrand expression representation:\n"
    s += "      %r\n" % integral._integrand
    s += "    Integrand expression short form:\n"
    s += "      %s" % integral._integrand
    return s

def form_info(form):
    ufl_assert(isinstance(form, Form), "Expecting a Form.")
    
    bf = basisfunctions(form)
    cf = coefficients(form)
    
    ci = form.cell_integrals()
    ei = form.exterior_facet_integrals()
    ii = form.interior_facet_integrals()
    
    s  = "Form info:\n"
    s += "  rank:                          %d\n" % len(bf)
    s += "  num_coefficients:              %d\n" % len(cf)
    s += "  num_cell_integrals:            %d\n" % len(ci)
    s += "  num_exterior_facet_integrals:  %d\n" % len(ei)
    s += "  num_interior_facet_integrals:  %d\n" % len(ii)
    
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
    return s

def _indent_string(n):
    return "    "*n

def _tree_format_expression(expression, indentation, parentheses):
    ind = _indent_string(indentation)
    if isinstance(expression, Terminal):
        s = ind + "%s" % repr(expression)
    else:
        sops = [_tree_format_expression(o, indentation+1, parentheses) for o in expression.operands()]
        s = ind + "%s\n" % type(expression).__name__ 
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
        ind = _indent_string(indentation)
        s += ind + "Form:\n"
        s += "\n".join(tree_format(itg, indentation+1, parentheses) for itg in chain(ci, ei, ii))
    
    elif isinstance(expression, Integral):
        ind = _indent_string(indentation)
        s += ind + "Integral:\n"
        ind = _indent_string(indentation+1)
        s += ind + "domain type: %s\n" % expression.domain_type()
        s += ind + "domain id: %d\n" % expression.domain_id()
        s += ind + "integrand:\n"
        s += tree_format(expression._integrand, indentation+2, parentheses)
    
    elif isinstance(expression, UFLObject):
        s += _tree_format_expression(expression, indentation, parentheses)
    
    else:
        ufl_error("Invalid object type %s" % type(expression))
    
    return s

