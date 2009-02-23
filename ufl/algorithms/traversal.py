"""This module contains algorithms for traversing expression trees in different ways."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-23"

# Modified by Anders Logg, 2008

from ufl.log import error
from ufl.assertions import ufl_assert
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

# Slow recursive version of traverse_terminals, kept here for illustration:
def __old_traverse_terminals(expr):
    if isinstance(expr, Terminal):
        yield expr
    else:
        for o in expr.operands():
            for t in traverse_terminals(o):
                yield t

# Faster (factor 10 or so) non-recursive version using a table instead of recursion (dynamic programming)
def traverse_terminals(expr):
    input = [expr]
    while input:
        e = input.pop()
        if isinstance(e, Terminal):
            yield e
        else:
            input.extend(e.operands())

def fast_pre_traversal(expr):
    """Yields o for each tree node o in expr, parent before child."""
    input = [expr]
    while input:
        l = input.pop()
        yield l
        input.extend(l.operands())

def pre_traversal(expr, stack=None):
    """Yields o for each tree node o in expr, parent before child.
    If a list is provided, the stack is updated while iterating."""
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")
    # yield parent
    yield expr
    # yield children
    if not isinstance(expr, Terminal):
        if stack is not None:
            stack.append(expr)
        for o in expr.operands():
            for i in pre_traversal(o, stack):
                yield i
        if stack is not None:
            stack.pop()

def post_traversal(expr, stack=None):
    """Yields o for each tree node o in expr, parent after child.
    If a list is provided, the stack is updated while iterating."""
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")
    # yield children
    if stack is not None:
        stack.append(expr)
    for o in expr.operands():
        for i in post_traversal(o, stack):
            yield i
    if stack is not None:
        stack.pop()
    # yield parent
    yield expr

def pre_walk(a, func):
    """Call func on each expression tree node in a, parent before child.
    The argument a can be a Form, Integral or Expr."""
    for e in iter_expressions(a):
        for o in pre_traversal(e):
            func(o)

def post_walk(a, func):
    """Call func on each expression tree node in a, parent after child.
    The argument a can be a Form, Integral or Expr."""
    for e in iter_expressions(a):
        for o in post_traversal(e):
            func(o)

def _walk(expr, pre_func, post_func, stack):
    # visit parent on the way in
    pre_func(expr, stack)
    # visit children
    stack.append(expr)
    for o in expr.operands():
        _walk(o, pre_func, post_func, stack)
    stack.pop()
    # visit parent on the way out
    post_func(expr, stack)

def walk(a, pre_func, post_func, stack=None):
    """Call pre_func and post_func on each expression tree node in a.
    
    The functions are called on a node before and 
    after its children are visited respectively.
    
    The argument a can be a Form, Integral or Expr."""
    if stack is None:
        stack = []
    for e in iter_expressions(a):
        _walk(e, pre_func, post_func, stack)

