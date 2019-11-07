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


def pre_traversal(expr):
    """Yield ``o`` for each tree node ``o`` in *expr*, parent before child."""
    lifo = [expr]
    while lifo:
        expr = lifo.pop()
        yield expr
        for op in expr.ufl_operands:
            lifo.append(op)


def post_traversal(expr):
    """Yield ``o`` for each node ``o`` in *expr*, child before parent."""
    lifo = [(expr, list(reversed(expr.ufl_operands)))]
    while lifo:
        expr, deps = lifo[-1]
        for i, dep in enumerate(deps):
            if dep is not None:
                lifo.append((dep, list(reversed(dep.ufl_operands))))
                deps[i] = None
                break
        else:
            yield expr
            lifo.pop()


def cutoff_post_traversal(expr, cutofftypes):
    """Yield ``o`` for each node ``o`` in *expr*, child before parent, but
    skipping subtrees of the cutofftypes."""
    lifo = [(expr, list(reversed(expr.ufl_operands)))]
    while lifo:
        expr, deps = lifo[-1]
        if cutofftypes[expr._ufl_typecode_]:
            yield expr
            lifo.pop()
        else:
            for i, dep in enumerate(deps):
                if dep is not None:
                    lifo.append((dep, list(reversed(dep.ufl_operands))))
                    deps[i] = None
                    break
            else:
                yield expr
                lifo.pop()


def unique_pre_traversal(expr, visited=None):
    """Yield ``o`` for each tree node ``o`` in *expr*, parent before child.

    This version only visits each node once.
    """
    if visited is None:
        visited = set()
    lifo = [expr]
    visited.add(expr)

    while lifo:
        expr = lifo.pop()
        yield expr
        for op in expr.ufl_operands:
            if op not in visited:
                lifo.append(op)
                visited.add(op)


def unique_post_traversal(expr, visited=None):
    """Yield ``o`` for each node ``o`` in *expr*, child before parent.

    Never visit a node twice."""
    lifo = [(expr, list(expr.ufl_operands))]
    if visited is None:
        visited = set()
    visited.add(expr)
    while lifo:
        expr, deps = lifo[-1]
        for i, dep in enumerate(deps):
            if dep is not None and dep not in visited:
                lifo.append((dep, list(dep.ufl_operands)))
                deps[i] = None
                break
        else:
            yield expr
            visited.add(expr)
            lifo.pop()


def cutoff_unique_post_traversal(expr, cutofftypes, visited=None):
    """Yield ``o`` for each node ``o`` in *expr*, child before parent.

    Never visit a node twice."""
    lifo = [(expr, list(reversed(expr.ufl_operands)))]
    if visited is None:
        visited = set()
    while lifo:
        expr, deps = lifo[-1]
        if cutofftypes[expr._ufl_typecode_]:
            yield expr
            visited.add(expr)
            lifo.pop()
        else:
            for i, dep in enumerate(deps):
                if dep is not None and dep not in visited:
                    lifo.append((dep, list(reversed(dep.ufl_operands))))
                    deps[i] = None
                    break
            else:
                yield expr
                visited.add(expr)
                lifo.pop()


def traverse_terminals(expr):
    for op in pre_traversal(expr):
        if op._ufl_is_terminal_:
            yield op


def traverse_unique_terminals(expr, visited=None):
    for op in unique_pre_traversal(expr, visited=visited):
        if op._ufl_is_terminal_:
            yield op
