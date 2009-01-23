"""This module contains algorithms for traversing expression trees in different ways."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-01-05"

# Modified by Anders Logg, 2008

from ufl.log import ufl_assert, error
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.integral import Integral
from ufl.form import Form
from ufl.variable import Variable

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
    elif isinstance(a, Expr):
        return (a,)
    error("Not an UFL type: %s" % str(type(a)))

def traverse_terminals(expr):
    if isinstance(expr, Terminal):
        yield expr
    else:
        for o in expr.operands():
            for t in traverse_terminals(o):
                yield t

def pre_traversal(expr, stack=None):
    "Yields (o, stack) for each tree node o in expr, parent before child."
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")
    if stack is None:
        stack = []
    # yield parent
    yield (expr, stack)
    # yield children
    if not isinstance(expr, Terminal):
        stack.append(expr)
        for o in expr.operands():
            for (i, dummy) in pre_traversal(o, stack):
                yield (i, stack)
        stack.pop()

def post_traversal(expr, stack=None):
    "Yields (o, stack) for each tree node o in expr, parent after child."
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")
    if stack is None:
        stack = []
    # yield children
    stack.append(expr)
    for o in expr.operands():
        for (i, dummy) in post_traversal(o, stack):
            yield (i, stack)
    stack.pop()
    # yield parent
    yield (expr, stack)

def pre_walk(a, func):
    """Call func on each expression tree node in a, parent before child.
    The argument a can be a Form, Integral or Expr."""
    for e in iter_expressions(a):
        for (o, stack) in pre_traversal(e, None):
            func(o)

def post_walk(a, func):
    """Call func on each expression tree node in a, parent after child.
    The argument a can be a Form, Integral or Expr."""
    for e in iter_expressions(a):
        for (o, stack) in post_traversal(e, None):
            func(o)

def _walk(expr, pre_func, post_func, stack=None):
    if stack is None:
        stack = []
    # visit parent on the way in
    pre_func(expr, stack)
    # visit children
    stack.append(expr)
    for o in expr.operands():
        _walk(o, pre_func, post_traversal, stack)
    stack.pop()
    # visit parent on the way out
    post_func(expr, stack)

def walk(a, pre_func, post_func):
    """Call pre_func and post_func on each expression tree node in a.
    
    The functions are called on a node before and 
    after its children are visited respectively.
    
    The argument a can be a Form, Integral or Expr."""
    for e in iter_expressions(a):
        _walk(e, pre_func, post_func)

