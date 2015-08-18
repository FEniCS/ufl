# -*- coding: utf-8 -*-
"""Various expression traversal utilities.

The algorithms here are non-recursive, which is faster than recursion
by a factor 10 or so because of the function call overhead.
"""

# Copyright (C) 2008-2014 Martin Sandve Alnes
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


# This limits the _depth_ of expression trees
_recursion_limit_ = 6400 # should be enough for everyone


def pre_traversal(expr):
    """Yields o for each tree node o in expr, parent before child."""
    stack = [None]*_recursion_limit_
    stack[0] = expr
    stacksize = 1
    while stacksize > 0:
        stacksize -= 1
        expr = stack[stacksize]
        yield expr
        for op in expr.ufl_operands:
            stack[stacksize] = op
            stacksize += 1


def post_traversal(expr):
    """Yields o for each node o in expr, child before parent."""
    stack = [None]*_recursion_limit_
    stacksize = 0

    ops = expr.ufl_operands
    stack[stacksize] = [expr, ops, len(ops)]
    stacksize += 1

    while stacksize > 0:
        entry = stack[stacksize - 1]
        if entry[2] == 0:
            yield entry[0]
            stacksize -= 1
        else:
            entry[2] -= 1
            o = entry[1][entry[2]]
            oops = o.ufl_operands
            stack[stacksize] = [o, oops, len(oops)]
            stacksize += 1


def cutoff_post_traversal(expr, cutofftypes):
    """Yields o for each node o in expr, child before parent, but skipping subtrees of the cutofftypes."""
    stack = [None]*_recursion_limit_
    stacksize = 0

    ops = expr.ufl_operands
    stack[stacksize] = [expr, ops, len(ops)]
    stacksize += 1

    while stacksize > 0:
        entry = stack[stacksize - 1]
        expr = entry[0]
        if entry[2] == 0 or cutofftypes[expr._ufl_typecode_]:
            yield expr
            stacksize -= 1
        else:
            entry[2] -= 1
            o = entry[1][entry[2]]
            if cutofftypes[expr._ufl_typecode_]:
                oops = ()
            else:
                oops = o.ufl_operands
            stack[stacksize] = [o, oops, len(oops)]
            stacksize += 1


def unique_pre_traversal(expr, visited=None):
    """Yields o for each tree node o in expr, parent before child.

    This version only visits each node once!
    """
    stack = [None]*_recursion_limit_
    stack[0] = expr
    stacksize = 1
    visited = visited or set()
    while stacksize > 0:
        stacksize -= 1
        expr = stack[stacksize]
        if expr not in visited:
            visited.add(expr)
            yield expr
            for op in expr.ufl_operands:
                stack[stacksize] = op
                stacksize += 1


def unique_post_traversal(expr, visited=None):
    """Yields o for each node o in expr, child before parent.

    Never visits a node twice."""
    stack = [None]*_recursion_limit_
    stack[0] = (expr, list(expr.ufl_operands))
    stacksize = 1
    visited = visited or set()
    while stacksize > 0:
        expr, ops = stack[stacksize - 1]
        for i, o in enumerate(ops):
            if o is not None and o not in visited:
                stack[stacksize] = (o, list(o.ufl_operands))
                stacksize += 1
                ops[i] = None
                break
        else:
            yield expr
            visited.add(expr)
            stacksize -= 1


def traverse_terminals(expr):
    "Iterate over all terminal objects in expression, including duplicates."
    stack = [None]*_recursion_limit_
    stack[0] = expr
    stacksize = 1
    while stacksize > 0:
        stacksize -= 1
        expr = stack[stacksize]
        if expr._ufl_is_terminal_:
            yield expr
        else:
            for op in expr.ufl_operands:
                stack[stacksize] = op
                stacksize += 1


def traverse_unique_terminals(expr):
    "Iterate over all terminal objects in expression, not including duplicates."
    stack = [None]*_recursion_limit_
    stack[0] = expr
    stacksize = 1
    visited = set()
    while stacksize > 0:
        stacksize -= 1
        expr = stack[stacksize]
        if expr not in visited:
            visited.add(expr)
            if expr._ufl_is_terminal_:
                yield expr
            else:
                for op in expr.ufl_operands:
                    stack[stacksize] = op
                    stacksize += 1
