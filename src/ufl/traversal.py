#!/usr/bin/env python

"""
This module contains algorithms for traversing expression trees, mostly using
generators and a kind of functional programming.

(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase).
"""

__version__ = "0.1"
__authors__ = "Martin Sandve Alnes"
__copyright__ = __authors__ + " (2008)"
__licence__ = "GPL" # TODO: which licence?
__date__ = "March 7th 2008"

from base import *


### Traversal utilities

def iter_depth_first(u):
    """Yields o for all nodes o in expression tree u, depth first."""
    for o in u.operands():
        for i in iter_depth_first(o):
            yield i
    yield u

def iter_width_first(u):
    """Yields o for all nodes o in expression tree u, width first."""
    yield u
    for o in u.operands():
        for i in iter_width_first(o):
            yield i

def traverse_depth_first(u, func):
    """Call func(o) for all nodes o in expression tree u, depth first."""
    for o in u.operands():
        traverse_depth_first(o, func)
    func(u)

def traverse_width_first(u, func):
    """Call func(o) for all nodes o in expression tree u, width first."""
    func(u)
    for o in u.operands():
        traverse_width_first(o, func)

def walk(u, func):
    traverse_width_first(u, func)


### Utilities for iteration over particular types

def iter_ufl_objs(u):
    """Utility function to handle Form, Integral and any UFLObject the same way.
       Returns an iterable over all Integral objects of u or the UFLObject u."""
    if isinstance(u, Form):
        objs = (itg.integrand for itg in u.integrals)
    elif isinstance(u, Integral):
        objs = (u.integrand,)
    else:
        objs = (u,)
    return objs

def iter_classes(u):
    """Returns an iterator over the unique classes used by objects in this expression."""
    returned = set()
    for o in iter_ufl_objs(u):
        for u in iter_depth_first(o):
            t = u.__class__
            if not t in returned:
                returned.add(t)
                yield t

def iter_elements(u):
    """Returns an iterator over the unique finite elements used in this form or expression."""
    returned = set()
    for o in iter_ufl_objs(u):
        for u in iter_depth_first(o):
            if isinstance(u, (BasisFunction, UFLCoefficient)) and not repr(u.element) in returned:
                returned.add(repr(u.element))
                yield u.element

def iter_basisfunctions(u):
    """Returns an iterator over the unique basis functions used in this form or expression."""
    returned = set()
    for o in iter_ufl_objs(u):
        for u in iter_depth_first(o):
            if isinstance(u, BasisFunction) and not repr(u) in returned:
                returned.add(repr(u))
                yield u

def iter_coefficients(u):
    """Returns an iterator over the unique coefficient functions used in this form or expression."""
    returned = set()
    for o in iter_ufl_objs(u):
        for u in iter_depth_first(o):
            if isinstance(u, UFLCoefficient) and not repr(u) in returned:
                returned.add(repr(u))
                yield u


