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

def basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique basisfunctions
    s = set()
    def func(o):
        if isinstance(o, BasisFunction):
            s.add(o)
    walk(a, func)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def coefficients(a):
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
    return [f.element for f in chain(basisfunctions(a), coefficients(a))]

def unique_elements(a):
    """Returns a sorted list of all elements used in a."""
    elements = set()
    for f in chain(iter_basisfunctions(a), iter_coefficients(a)):
        elements.add(f.element)
    return elements

def unique_classes(a):
    """Returns a set of all unique UFLObject subclasses used in a."""
    classes = set()
    for e in iter_expressions(a):
        for o in iter_depth_first(e):
            classes.add(o.__class__)
    return classes

def duplications(u):
    """Returns a ??? of all repeated expressions in u."""
    from visitor import SubtreeFinder
    vis = SubtreeFinder()
    vis.visit(f)
    return vis.duplicated


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
    # FIXME: Implement (like FFCs simplify done by Marie)

