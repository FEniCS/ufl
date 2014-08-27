"""Various expression traversal utilities.

The algorithms here are non-recursive, which is both faster
than recursion by a factor 10 or so because of the function
call overhead, and avoids the finite recursive call limit.
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


def pre_traversal(expr):
    """Yields o for each tree node o in expr, parent before child."""
    stackcap = 1000
    stack = [None]*stackcap
    stack[0] = expr
    stacksize = 1
    while stacksize > 0:
        expr = stack[stacksize-1]
        stacksize -= 1
        yield expr
        for op in expr.ufl_operands:
            stack[stacksize] = op
            stacksize += 1


def post_traversal(expr):
    """Yields o for each node o in expr, child before parent."""
    stackcap = 1000
    stack = [None]*stackcap
    stack[0] = (expr, list(expr.ufl_operands))
    stacksize = 1
    while stacksize > 0:
        expr, ops = stack[stacksize-1]
        if len(ops) == 0:
            yield expr
            stacksize -= 1
        else:
            o = ops.pop()
            if stacksize >= stackcap:
                stack.append(None)
                stackcap += 1
            stack[stacksize] = (o, list(o.ufl_operands))
            stacksize += 1


def cutoff_post_traversal(expr, cutofftypes):
    """Yields o for each node o in expr, child before parent, but skipping subtrees of the cutofftypes."""
    stackcap = 1000
    stack = [None]*stackcap
    stack[0] = (expr, list(expr.ufl_operands))
    stacksize = 1
    while stacksize > 0:
        expr, ops = stack[stacksize-1]
        if len(ops) == 0 or cutofftypes[expr._ufl_typecode_]:
            yield expr
            stacksize -= 1
        else:
            o = ops.pop()
            if stacksize >= stackcap:
                stack.append(None)
                stackcap += 1
            stack[stacksize] = (o, list(o.ufl_operands))
            stacksize += 1


# TODO: Apply faster manual stack handling to below algorithms like above (becomes a bit uglier but is faster)


def unique_pre_traversal(expr, visited=None):
    """Yields o for each tree node o in expr, parent before child.

    This version only visits each node once!
    """
    stack = [expr]
    visited = visited or set()
    while stack:
        expr = stack.pop()
        if expr not in visited:
            visited.add(expr)
            yield expr
            stack.extend(expr.ufl_operands)


def unique_post_traversal(expr, visited=None):
    """Yields o for each node o in expr, child before parent.

    Never visits a node twice."""
    stack = []
    stack.append((expr, list(expr.ufl_operands)))
    visited = visited or set()
    while stack:
        expr, ops = stack[-1]
        for i, o in enumerate(ops):
            if o is not None and o not in visited:
                stack.append((o, list(o.ufl_operands)))
                ops[i] = None
                break
        else:
            yield expr
            visited.add(expr)
            stack.pop()


def traverse_terminals(expr):
    "Iterate over all terminal objects in expression, including duplicates."
    stack = [expr]
    while stack:
        e = stack.pop()
        if e._ufl_is_terminal_:
            yield e
        else:
            stack.extend(e.ufl_operands)


def traverse_unique_terminals(expr):
    "Iterate over all terminal objects in expression, not including duplicates."
    stack = [expr]
    visited = set()
    while stack:
        e = stack.pop()
        if e not in visited:
            visited.add(e)
            if e._ufl_is_terminal_:
                yield e
            else:
                stack.extend(e.ufl_operands)
