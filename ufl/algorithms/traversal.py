"""This module contains algorithms for traversing expression trees in different ways."""

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008
#
# First added:  2008-03-14
# Last changed: 2011-06-02

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.integral import Integral
from ufl.form import Form

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
        return (itg._integrand for itg in a.integrals())
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

def traverse_terminals2(expr, visited=None):
    input = [expr]
    visited = visited or set()
    while input:
        e = input.pop()
        if e not in visited:
            visited.add(e)
            if isinstance(e, Terminal):
                yield e
            else:
                input.extend(e.operands())

def traverse_operands(expr):
    input = [expr]
    while input:
        e = input.pop()
        if not isinstance(e, Terminal):
            yield e
            input.extend(e.operands())

# Moved to common because it is without dependencies and this avoids circular deps
from ufl.common import fast_pre_traversal, fast_post_traversal

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

