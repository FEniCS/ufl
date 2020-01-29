# -*- coding: utf-8 -*-
"""This module provides utility functions for computing permutations
and generating index lists."""

# Copyright (C) 2008-2016 Anders Logg and Kent-Andre Mardal
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Martin Alnæs 2009-2016


def compute_indices(shape):
    "Compute all index combinations for given shape"
    if len(shape) == 0:
        return ((),)
    sub_indices = compute_indices(shape[1:])
    indices = []
    for i in range(shape[0]):
        for sub_index in sub_indices:
            indices.append((i,) + sub_index)
    return tuple(indices)


# functional version:
def compute_indices2(shape):
    "Compute all index combinations for given shape"
    return ((),) if len(shape) == 0 else tuple((i,) + sub_index for i in range(shape[0]) for sub_index in compute_indices2(shape[1:]))


def build_component_numbering(shape, symmetry):
    """Build a numbering of components within the given value shape,
    taking into consideration a symmetry mapping which leaves the
    mapping noncontiguous. Returns a dict { component -> numbering }
    and an ordered list of components [ numbering -> component ].  The
    dict contains all components while the list only contains the ones
    not mapped by the symmetry mapping.

    """
    vi2si, si2vi = {}, []
    indices = compute_indices(shape)
    # Number components not in symmetry mapping
    for c in indices:
        if c not in symmetry:
            vi2si[c] = len(si2vi)
            si2vi.append(c)
    # Copy numbering to mapped components
    for c in indices:
        if c in symmetry:
            vi2si[c] = vi2si[symmetry[c]]
    # Validate
    for k, c in enumerate(si2vi):
        assert vi2si[c] == k
    return vi2si, si2vi


def compute_permutations(k, n, skip=None):
    """Compute all permutations of k elements from (0, n) in rising order.
    Any elements that are contained in the list skip are not included.

    """
    if k == 0:
        return []
    if skip is None:
        skip = []
    if k == 1:
        return [(i,) for i in range(n) if i not in skip]
    pp = compute_permutations(k - 1, n, skip)
    permutations = []
    for i in range(n):
        if i in skip:
            continue
        for p in pp:
            if i < p[0]:
                permutations.append((i,) + p)
    return permutations


def compute_permutation_pairs(j, k):
    """Compute all permutations of j + k elements from (0, j + k) in
    rising order within (0, j) and (j, j + k) respectively.

    """
    permutations = []
    pp0 = compute_permutations(j, j + k)
    for p0 in pp0:
        pp1 = compute_permutations(k, j + k, p0)
        for p1 in pp1:
            permutations.append((p0, p1))
    return permutations


def compute_sign(permutation):
    "Compute sign by sorting."
    sign = 1
    n = len(permutation)
    p = [p for p in permutation]
    for i in range(n - 1):
        for j in range(n - 1):
            if p[j] > p[j + 1]:
                (p[j], p[j + 1]) = (p[j + 1], p[j])
                sign = -sign
            elif p[j] == p[j + 1]:
                return 0
    return sign


def compute_order_tuples(k, n):
    "Compute all tuples of n integers such that the sum is k"
    if n == 1:
        return ((k,),)
    order_tuples = []
    for i in range(k + 1):
        for order_tuple in compute_order_tuples(k - i, n - 1):
            order_tuples.append(order_tuple + (i,))
    return tuple(order_tuples)
