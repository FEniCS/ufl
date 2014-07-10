"Various expression traversal utilities."

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

def fast_pre_traversal(expr):
    """Yields o for each tree node o in expr, parent before child."""
    input = [expr]
    while input:
        l = input.pop()
        yield l
        input.extend(l.operands())

def unique_pre_traversal(expr, visited=None):
    """Yields o for each tree node o in expr, parent before child.

    This version only visits each node once!
    """
    input = [expr]
    visited = visited or set()
    while input:
        l = input.pop()
        if l not in visited:
            visited.add(l)
            yield l
            input.extend(l.operands())
fast_pre_traversal2 = unique_pre_traversal # TODO: Remove

def unique_post_traversal(expr, visited=None):
    """Yields o for each node o in expr, child before parent.

    Never visits a node twice."""
    stack = []
    stack.append((expr, list(expr.operands())))
    visited = visited or set()
    while stack:
        expr, ops = stack[-1]
        for i, o in enumerate(ops):
            if o is not None and o not in visited:
                stack.append((o, list(o.operands())))
                ops[i] = None
                break
        else:
            yield expr
            visited.add(expr)
            stack.pop()

def fast_post_traversal2(expr, visited=None):
    """Yields o for each tree node o in expr, child before parent."""
    stack = [expr]
    visited = visited or set()
    while stack:
        curr = stack[-1]
        for o in curr.operands():
            if o not in visited:
                stack.append(o)
                break
        else:
            yield curr
            visited.add(curr)
            stack.pop()

def fast_post_traversal(expr): # TODO: Would a non-recursive implementation save anything here?
    """Yields o for each tree node o in expr, child before parent."""
    # yield children
    for o in expr.operands():
        for i in fast_post_traversal(o):
            yield i
    # yield parent
    yield expr
