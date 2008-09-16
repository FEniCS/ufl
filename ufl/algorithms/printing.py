"""A collection of utility algorithms for printing
of UFL objects, mostly intended for debugging purposers."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-16"

from itertools import chain

from ..output import ufl_assert
from ..form import Form

#--- Utilities for constructing informative strings from UFL objects ---

def integral_info(integral):
    s  = "  Integral over %s domain %d:\n" % (integral._domain_type, integral._domain_id)
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

