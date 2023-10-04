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


def build_component_numbering(shape, symmetry):
    """Build a numbering of components within the given value shape.

    This takes into consideration a symmetry mapping which leaves the
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
