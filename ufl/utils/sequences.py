"""Various sequence manipulation utilities."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from functools import reduce

import numpy as np


def product(sequence):
    """Return the product of all elements in a sequence."""
    p = 1
    for f in sequence:
        p *= f
    return p


def max_degree(degrees):
    """Maximum degree for mixture of scalar and tuple degrees."""
    # np.maximum broadcasts scalar degrees to tuple degrees if
    # necessary.  reduce applies np.maximum pairwise.
    degree = reduce(np.maximum, map(np.asarray, degrees))
    if degree.ndim:
        degree = tuple(map(int, degree))  # tuple degree
    else:
        degree = int(degree)  # scalar degree
    return degree
