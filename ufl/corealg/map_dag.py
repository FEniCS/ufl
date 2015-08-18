# -*- coding: utf-8 -*-
"""Basic algorithms for applying functions to subexpressions."""

# Copyright (C) 2014 Martin Sandve Alnes
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

from ufl.core.expr import Expr
from ufl.corealg.traversal import post_traversal, cutoff_post_traversal
from ufl.corealg.multifunction import MultiFunction

def map_expr_dag(function, expression, compress=True):
    """Apply a function to each subexpression node in expression dag.

    If compress is True (default), the output object from
    the function is cached in a dict and reused such that the
    resulting expression dag does not contain duplicate objects.

    Returns the result of the final function call.
    """

    # Temporary data structures
    vcache = {}
    rcache = {}
    results = []

    # Build mapping typecode:bool, for which types to skip the subtree of
    if isinstance(function, MultiFunction):
        cutoff_types = function._is_cutoff_type
    else:
        # Regular function: no skipping supported
        cutoff_types = [False]*Expr._ufl_num_typecodes_

    # Pick faster traversal algorithm if we have no cutoffs
    if any(cutoff_types):
        traversal = lambda expression: cutoff_post_traversal(expression, cutoff_types)
    else:
        traversal = lambda expression: post_traversal(expression)

    # Iterate over all subexpression nodes, child before parent
    for v in traversal(expression):

        # Check if v is in vcache (to be able to skip transformations)
        i = vcache.get(v)

        # Cache hit: skip transformation
        if i is not None:
            continue

        # Cache miss: Get transformed operands, then apply transformation
        if cutoff_types[v._ufl_typecode_]:
            r = function(v)
        else:
            rops = [results[vcache[u]] for u in v.ufl_operands]
            r = function(v, *rops)

        # Optionally check if r is in rcache (to be able to keep representation of result compact)
        i = rcache.get(r) if compress else None

        if i is None:
            # Cache miss: Assign result index and store in results list
            i = len(results)
            results.append(r)
            # Store in rcache
            if compress:
                rcache[r] = i

        # Store in vcache
        vcache[v] = i

    return results[i]
