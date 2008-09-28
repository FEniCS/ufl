"""This module contains algorithms for traversing expression trees, mostly using
generators and a kind of functional programming.

(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase)."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-28"

from ..output import ufl_assert
from ..base import UFLObject
from ..integral import Integral
from ..form import Form


#--- Traversal utilities ---

def iter_expressions(a):
    """Utility function to handle Form, Integral and any UFLObject
    the same way when inspecting expressions.
    Returns an iterable over UFLObject instances:
    - a is an UFLObject: (a,)
    - a is an Integral:  the integrand expression of a
    - a is a  Form:      all integrand expressions of all integrals
    """
    if isinstance(a, Form):
        return (itg._integrand for itg in a._integrals)
    elif isinstance(a, Integral):
        return (a._integrand,)
    else:
        return (a,)

def post_traversal(expression):
    "Yields o for each tree node o in expression, child before parent."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject.")
    for o in expression.operands():
        for i in post_traversal(o):
            yield i
    yield expression

def pre_traversal(expression):
    "Yields o for each tree node o in expression, parent before child."""
    yield expression
    for o in expression.operands():
        for i in pre_traversal(o):
            yield i

def post_walk(a, func):
    """Call func on each expression tree node in a, child before parent.
    The argument a can be a Form, Integral or UFLObject."""
    for e in iter_expressions(a):
        for o in post_traversal(e):
            func(o)

def pre_walk(a, func):
    """Call func on each expression tree node in a, parent before child.
    The argument a can be a Form, Integral or UFLObject."""
    for e in iter_expressions(a):
        for o in pre_traversal(e):
            func(o)

def walk(a, func):
    """Call func on each expression tree node in a.
    The argument a can be a Form, Integral or UFLObject."""
    pre_walk(a, func)

