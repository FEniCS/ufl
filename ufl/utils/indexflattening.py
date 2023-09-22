"""This module contains a collection of utilities for mapping between multiindices and a flattened index space."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


def shape_to_strides(sh):
    """Return a tuple of strides given a shape tuple."""
    n = len(sh)
    if not n:
        return ()
    strides = [None] * n
    strides[n - 1] = 1
    for i in range(n - 1, 0, -1):
        strides[i - 1] = strides[i] * sh[i]
    return tuple(strides)


def flatten_multiindex(ii, strides):
    """Return the flat index corresponding to the given multiindex."""
    i = 0
    for c, s in zip(ii, strides):
        i += c * s
    return i


def unflatten_index(i, strides):
    """Return the multiindex corresponding to the given flat index."""
    ii = []
    for s in strides:
        ii.append(i // s)
        i %= s
    return tuple(ii)
