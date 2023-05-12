# -*- coding: utf-8 -*-
"""Basic algorithms for applying functions to subexpressions."""

# Copyright (C) 2014-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

from ufl_legacy.core.expr import Expr
from ufl_legacy.corealg.traversal import unique_post_traversal, cutoff_unique_post_traversal
from ufl_legacy.corealg.multifunction import MultiFunction


def map_expr_dag(function, expression,
                 compress=True,
                 vcache=None,
                 rcache=None):
    """Apply a function to each subexpression node in an expression DAG.

    If *compress* is ``True`` (default) the output object from
    the function is cached in a ``dict`` and reused such that the
    resulting expression DAG does not contain duplicate objects.

    If the same funtion is called multiple times in a transformation
    (as for example in apply_derivatives), then to reuse caches across
    the call, provide these two arguments:

    :arg vcache: Optional dict for caching results of intermediate transformations
    :arg rcache: Optional dict for caching results for compression.

    Return the result of the final function call.
    """
    result, = map_expr_dags(function, [expression], compress=compress,
                            vcache=vcache,
                            rcache=rcache)
    return result


def map_expr_dags(function, expressions,
                  compress=True,
                  vcache=None,
                  rcache=None):
    """Apply a function to each subexpression node in an expression DAG.

    If *compress* is ``True`` (default) the output object from
    the function is cached in a ``dict`` and reused such that the
    resulting expression DAG does not contain duplicate objects.

    If the same funtion is called multiple times in a transformation
    (as for example in apply_derivatives), then to reuse caches across
    the call, provide these two arguments:

    :arg vcache: Optional dict for caching results of intermediate transformations
    :arg rcache: Optional dict for caching results for compression.

    Return a list with the result of the final function call for each expression.
    """

    # Temporary data structures
    # expr -> r = function(expr,...),  cache of intermediate results
    vcache = {} if vcache is None else vcache
    # r -> r,  cache of result objects for memory reuse
    rcache = {} if rcache is None else rcache

    # Build mapping typecode:bool, for which types to skip the subtree of
    if isinstance(function, MultiFunction):
        cutoff_types = function._is_cutoff_type
        handlers = function._handlers  # Optimization
    else:
        # Regular function: no skipping supported
        cutoff_types = [False] * Expr._ufl_num_typecodes_
        handlers = [function] * Expr._ufl_num_typecodes_

    # Create visited set here to share between traversal calls
    visited = set()

    # Pick faster traversal algorithm if we have no cutoffs
    if any(cutoff_types):
        def traversal(expression):
            return cutoff_unique_post_traversal(expression, cutoff_types, visited)
    else:
        def traversal(expression):
            return unique_post_traversal(expression, visited)

    for expression in expressions:
        # Iterate over all subexpression nodes, child before parent
        for v in traversal(expression):
            # Skip transformations on cache hit
            if v in vcache:
                continue

            # Cache miss: Get transformed operands, then apply transformation
            if cutoff_types[v._ufl_typecode_]:
                r = handlers[v._ufl_typecode_](v)
            else:
                r = handlers[v._ufl_typecode_](v, *[vcache[u] for u in v.ufl_operands])

            # Optionally check if r is in rcache, a memory optimization
            # to be able to keep representation of result compact
            if compress:
                r2 = rcache.get(r)
                if r2 is None:
                    # Cache miss: store in rcache
                    rcache[r] = r
                else:
                    # Cache hit: Use previously computed object r2,
                    # allowing r to be garbage collected as soon as possible
                    r = r2

            # Store result in cache
            vcache[v] = r

    return [vcache[expression] for expression in expressions]
