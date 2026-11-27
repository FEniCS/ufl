"""This module provides utility functions for computing permutations and generating index lists."""
# Copyright (C) 2008-2016 Anders Logg and Kent-Andre Mardal
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Martin Alnæs 2009-2016


def compute_indices(shape):
    """Compute all index combinations for given shape."""
    if len(shape) == 0:
        return ((),)
    sub_indices = compute_indices(shape[1:])
    indices = []
    for i in range(shape[0]):
        for sub_index in sub_indices:
            indices.append((i,) + sub_index)
    return tuple(indices)
