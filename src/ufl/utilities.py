#!/usr/bin/env python

"""
Utility algorithms for inspection, conversion or transformation
of UFL objects in various ways.

(Currently, some utility functions are located in visitor.py,
some in traversal.py, and some in transformers.py,
depending on the method of implementation.
This file should contain userfriendly front-ends
to all the utility algorithms that we want to expose.)
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-04-02"

from itertools import chain

from base import *
from visitor import *
from transformers import *
from traversal import *


### Utilities to extract information from an expression:

def extract_type(a, ufl_type):
    """Returns a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or UFLObject.
    """
    iter = (o for e in iter_expressions(a) \
              for o in iter_child_first(e) \
              if isinstance(o, ufl_type) )
    return set(iter)

def basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique basisfunctions
    s = extract_type(a, BasisFunction)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique coefficients
    s = extract_type(a, Function)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

# alternative implementation:
def _coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Function):
            s.add(o)
    walk(a, func)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def elements(a):
    """Returns a sorted list of all elements used in a."""
    return [f._element for f in chain(basisfunctions(a), coefficients(a))]

def unique_elements(a):
    """Returns a sorted list of all elements used in a."""
    elements = set()
    for f in chain(basisfunctions(a), coefficients(a)):
        elements.add(f._element)
    return elements

def unique_classes(a):
    """Returns a set of all unique UFLObject subclasses used in a."""
    classes = set()
    for e in iter_expressions(a):
        for o in iter_child_first(e):
            classes.add(o.__class__)
    return classes

def variables(a):
    """Returns a set of all variables in a,
    which can be a Form, Integral or UFLObject.
    """
    return extract_type(a, Variable)

def duplications(a):
    """Returns a set of all repeated expressions in u."""
    handled = set()
    duplicated = set()
    for e in iter_expressions(a):
        for o in iter_parent_first(e):
            if o in handled:
                duplicated.add(o)
            handled.add(o)
    return duplicated

def integral_info(itg):
    s  = "  Integral over %s domain %d:\n" % (itg._domain_type, itg._domain_id)
    s += "    Integrand expression representation:\n"
    s += "      %s\n" % repr(itg._integrand)
    s += "    Integrand expression short form:\n"
    s += "      %s" % str(itg._integrand)
    return s

def form_info(a):
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    
    bf = basisfunctions(a)
    cf = coefficients(a)
    
    ci = a.cell_integrals()
    ei = a.exterior_facet_integrals()
    ii = a.interior_facet_integrals()
    
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

### Utilities to convert expression to a different form:

def flatten(u):
    "Flatten (a+b)+(c+d) into a (a+b+c+d) and (a*b)*(c*d) into (a*b*c*d)."
    vis = TreeFlattener()
    flat_u = vis.visit(f)
    return flat_u

def apply_summation(u):
    "Expand all repeated indices into explicit sums with fixed indices."
    ufl_error("Not implemented")
    # FIXME: Implement

def discover_indices(u):
    "Convert explicit sums into implicit sums (repeated indices)."
    ufl_error("Not implemented")
    # FIXME: Implement (like FFCs 'simplify' done by Marie)
