#!/usr/bin/env python

"""
This module contains algorithms for traversing expression trees, mostly using
generators and a kind of functional programming.

(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase).
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-04-01"

from all import *
from integral import *
from form import *


### Traversal utilities

def iter_expressions(u):
    """Utility function to handle Form, Integral and any UFLObject
    the same way when inspecting expressions.
    Returns an iterable over UFLObject instances:
    - u is an UFLObject: (u,)
    - u is an Integral:  the integrand expression of u
    - u is a  Form:      all integrand expressions of all integrals
    """
    if isinstance(u, Form):
        return (itg.integrand for itg in u.integrals)
    elif isinstance(u, Integral):
        return (u.integrand,)
    else:
        return (u,)

def iter_child_first(u):
    """Yields o for all nodes o in expression tree u, child before parent."""
    for o in u.operands():
        for i in iter_child_first(o):
            yield i
    yield u

def iter_parent_first(u):
    """Yields o for all nodes o in expression tree u, parent before child."""
    yield u
    for o in u.operands():
        for i in iter_parent_first(o):
            yield i

def traverse_child_first(u, func):
    """Call func(o) for all nodes o in expression tree u, child before parent."""
    for o in u.operands():
        traverse_child_first(o, func)
    func(u)

def traverse_parent_first(u, func):
    """Call func(o) for all nodes o in expression tree u, parent before child."""
    func(u)
    for o in u.operands():
        traverse_parent_first(o, func)

def walk(a, func):
    """Call func(o) for all nodes o in expression tree a."""
    for e in iter_expressions(a):
        traverse_parent_first(e, func)
