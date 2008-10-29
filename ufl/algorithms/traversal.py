"""This module contains algorithms for traversing expression trees, mostly using
generators and a kind of functional programming.

(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase)."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-23"

# Modified by Anders Logg, 2008

from ..output import ufl_assert
from ..base import Expr, Terminal
from ..integral import Integral
from ..form import Form
from ..variable import Variable

#--- Traversal utilities ---

def iter_expressions(a):
    """Utility function to handle Form, Integral and any Expr
    the same way when inspecting expressions.
    Returns an iterable over Expr instances:
    - a is an Expr: (a,)
    - a is an Integral:  the integrand expression of a
    - a is a  Form:      all integrand expressions of all integrals
    """
    if isinstance(a, Form):
        return (itg._integrand for itg in a._integrals)
    elif isinstance(a, Integral):
        return (a._integrand,)
    else:
        return (a,)

def pre_traversal(expression, stack=None, traverse_into_variables=True):
    "Yields (o, stack) for each tree node o in expression, parent before child."
    if stack is None: stack = []
    ufl_assert(isinstance(expression, Expr), "Expecting Expr.")
    # yield parent
    yield (expression, stack)
    # yield children
    if traverse_into_variables and isinstance(expression, Variable):
        stack.append(expression)
        for (i, dummy) in pre_traversal(expression._expression, stack):
            yield (i, stack)
        stack.pop()
    elif not isinstance(expression, Terminal):
        stack.append(expression)
        for o in expression.operands():
            for (i, dummy) in pre_traversal(o, stack):
                yield (i, stack)
        stack.pop()

def post_traversal(expression, stack=None, traverse_into_variables=True):
    "Yields (o, stack) for each tree node o in expression, parent after child."
    if stack is None: stack = []
    ufl_assert(isinstance(expression, Expr), "Expecting Expr.")
    # yield children
    if traverse_into_variables and isinstance(expression, Variable):
        stack.append(expression)
        for (i, dummy) in post_traversal(expression._expression, stack):
            yield (i, stack)
        stack.pop()
    elif not isinstance(expression, Terminal):
        stack.append(expression)
        for o in expression.operands():
            for (i, dummy) in post_traversal(o, stack):
                yield (i, stack)
        stack.pop()
    # yield parent
    yield (expression, stack)

def traversal(expression, stack=None):
    "Yields (o, stack) for each tree node o in expression."
    return pre_traversal(expression, stack)

def pre_walk(a, func, traverse_into_variables=True):
    """Call func on each expression tree node in a, parent before child.
    The argument a can be a Form, Integral or Expr."""
    for e in iter_expressions(a):
        for (o, stack) in pre_traversal(e, None, traverse_into_variables):
            func(o)

def post_walk(a, func, traverse_into_variables=True):
    """Call func on each expression tree node in a, parent after child.
    The argument a can be a Form, Integral or Expr."""
    for e in iter_expressions(a):
        for (o, stack) in post_traversal(e, None, traverse_into_variables):
            func(o)

def walk(a, func, traverse_into_variables=True):
    """Call func on each expression tree node in a.
    The argument a can be a Form, Integral or Expr."""
    pre_walk(a, func, traverse_into_variables)
