"""This module contains algorithms for traversing expression trees, mostly using
generators and a kind of functional programming.

(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase)."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-13"

from ..all import Integral, Form


#--- Traversal utilities ---

def iter_expressions(u):
    """Utility function to handle Form, Integral and any UFLObject
    the same way when inspecting expressions.
    Returns an iterable over UFLObject instances:
    - u is an UFLObject: (u,)
    - u is an Integral:  the integrand expression of u
    - u is a  Form:      all integrand expressions of all integrals
    """
    if isinstance(u, Form):
        return (itg._integrand for itg in u._integrals)
    elif isinstance(u, Integral):
        return (u._integrand,)
    else:
        return (u,)

def post_traversal(u):
    """Yields o for all nodes o in expression tree u, child before parent."""
    for o in u.operands():
        for i in post_traversal(o):
            yield i
    yield u

def pre_traversal(u):
    """Yields o for all nodes o in expression tree u, parent before child."""
    yield u
    for o in u.operands():
        for i in pre_traversal(o):
            yield i

def post_walk(a, func):
    for e in iter_expressions(a):
        for o in post_traversal(e):
            func(o)

def pre_walk(a, func):
    for e in iter_expressions(a):
        for o in pre_traversal(e):
            func(o)

def walk(a, func):
    pre_walk(a, func)

