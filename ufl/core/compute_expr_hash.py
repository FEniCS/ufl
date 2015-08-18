# -*- coding: utf-8 -*-
"""Non-recursive traversal based hash computation algorithm.

Fast iteration over nodes in an Expr DAG to compute
memoized hashes for all unique nodes.
"""

# Copyright (C) 2015 Martin Sandve Alnes
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


def compute_expr_hash(expr):
    """Compute hashes of expr and all its nodes efficiently without using Python recursion."""
    if expr._hash is not None:
        return expr._hash

    stack = [None]*_recursion_limit_
    stacksize = 0

    ops = expr.ufl_operands
    stack[stacksize] = [expr, ops, len(ops)]
    stacksize += 1

    while stacksize > 0:
        entry = stack[stacksize - 1]
        e = entry[0]
        if e._hash is not None:
            # cutoff: don't need to visit children when hash has previously been computed
            stacksize -= 1
        elif entry[2] == 0:
            # all children consumed: trigger memoized hash computation
            e._hash = e._ufl_compute_hash_()
            stacksize -= 1
        else:
            # add children to stack to hash them first
            entry[2] -= 1
            o = entry[1][entry[2]]
            oops = o.ufl_operands
            stack[stacksize] = [o, oops, len(oops)]
            stacksize += 1

    return expr._hash
