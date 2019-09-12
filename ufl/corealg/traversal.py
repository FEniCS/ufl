# -*- coding: utf-8 -*-
"""Various expression traversal utilities.

The algorithms here are non-recursive, which is faster than recursion
by a factor of 10 or so because of the function call overhead.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

# This limits the _depth_ of expression trees
_recursion_limit_ = 6400  # should be enough for everyone


def pre_traversal(expr):
    """Yield ``o`` for each tree node ``o`` in *expr*, parent before child."""
    stack = [None] * _recursion_limit_
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
    """Yield ``o`` for each node ``o`` in *expr*, child before parent."""
    stack = [None] * _recursion_limit_
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
    """Yield ``o`` for each node ``o`` in *expr*, child before parent, but
    skipping subtrees of the cutofftypes."""
    stack = [None] * _recursion_limit_
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
    """Yield ``o`` for each tree node ``o`` in *expr*, parent before child.

    This version only visits each node once.
    """
    stack = [None] * _recursion_limit_
    stack[0] = expr
    stacksize = 1
    if visited is None:
        visited = set()
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
    """Yield ``o`` for each node ``o`` in *expr*, child before parent.

    Never visit a node twice."""
    stack = [None] * _recursion_limit_
    stack[0] = (expr, list(expr.ufl_operands))
    stacksize = 1
    if visited is None:
        visited = set()
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


def cutoff_unique_post_traversal(expr, cutofftypes, visited=None):
    """Yield ``o`` for each node ``o`` in *expr*, child before parent.

    Never visit a node twice."""
    stack = [None] * _recursion_limit_
    stack[0] = (expr, () if cutofftypes[expr._ufl_typecode_] else list(expr.ufl_operands))
    stacksize = 1
    if visited is None:
        visited = set()
    while stacksize > 0:
        expr, ops = stack[stacksize - 1]
        for i, o in enumerate(ops):
            if o is not None and o not in visited:
                stack[stacksize] = (o, () if cutofftypes[o._ufl_typecode_] else list(o.ufl_operands))
                stacksize += 1
                ops[i] = None
                break
        else:
            yield expr
            visited.add(expr)
            stacksize -= 1


def traverse_terminals(expr):
    "Iterate over all terminal objects in *expr*, including duplicates."
    stack = [None] * _recursion_limit_
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


def traverse_unique_terminals(expr, visited=None):
    "Iterate over all terminal objects in *expr*, not including duplicates."
    stack = [None] * _recursion_limit_
    stack[0] = expr
    stacksize = 1
    if visited is None:
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
