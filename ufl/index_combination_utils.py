# -*- coding: utf-8 -*-
"Utilities for analysing and manipulating free index tuples"

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from six.moves import zip
from six.moves import xrange as range

from ufl.log import error
from ufl.core.multiindex import FixedIndex, Index, indices


# FIXME: Some of these might be merged into one function, some might
# be optimized

def unique_sorted_indices(indices):
    """Given a list of (id, dim) tuples already sorted by id,
    return a unique list with duplicates removed.
    Also checks that the dimensions of duplicates are matching.
    """
    newindices = []
    prev = (None, None)
    for i in indices:
        if i[0] != prev[0]:
            newindices.append(i)
            prev = i
        else:
            if i[1] != prev[1]:
                error("Nonmatching dimensions for free indices with same id!")
    return tuple(newindices)


def merge_unique_indices(afi, afid, bfi, bfid):
    """Merge two pairs of (index ids, index dimensions) sequences into one pair without duplicates.

    The id tuples afi, bfi are assumed already sorted by id.
    Given a list of (id, dim) tuples already sorted by id,
    return a unique list with duplicates removed.
    Also checks that the dimensions of duplicates are matching.
    """

    na = len(afi)
    nb = len(bfi)

    if na == 0:
        return bfi, bfid
    elif nb == 0:
        return afi, afid

    ak = 0
    bk = 0

    fi = []
    fid = []

    while True:
        if afi[ak] < bfi[bk]:
            fi.append(afi[ak])
            fid.append(afid[ak])
            ak += 1
        elif afi[ak] > bfi[bk]:
            fi.append(bfi[bk])
            fid.append(bfid[bk])
            bk += 1
        else:
            fi.append(afi[ak])
            fid.append(afid[ak])
            ak += 1
            bk += 1

        if ak == na:
            if bk != nb:
                fi.extend(bfi[bk:])
                fid.extend(bfid[bk:])
            break
        elif bk == nb:
            fi.extend(afi[ak:])
            fid.extend(afid[ak:])
            break

    return tuple(fi), tuple(fid)


def remove_indices(fi, fid, rfi):
    """
    """
    if not rfi:
        return fi, fid

    rfip = sorted((r, p) for p, r in enumerate(rfi))

    nrfi = len(rfi)
    nfi = len(fi)
    shape = [None]*nrfi
    k = 0
    pos = 0
    newfiid = []
    while pos < nfi:
        rk = rfip[k][0]

        # Keep
        while fi[pos] < rk:
            newfiid.append((fi[pos], fid[pos]))
            pos += 1

        # Skip
        removed = 0
        while pos < nfi and fi[pos] == rk:
            shape[rfip[k][1]] = fid[pos]
            pos += 1
            removed += 1

        # Expecting to find each index from rfi in fi
        if not removed:
            error("Index to be removed ({0}) not part of indices ({1}).".format(rk, fi))

        # Next to remove
        k += 1
        if k == nrfi:
            # No more to remove, keep the rest
            if pos < nfi:
                newfiid.extend(zip(fi[pos:], fid[pos:]))
            break

    assert None not in shape

    # Unpack into two tuples
    fi, fid = zip(*newfiid) if newfiid else ((), ())

    return fi, fid, tuple(shape)


def create_slice_indices(component, shape, fi):
    all_indices = []
    slice_indices = []
    repeated_indices = []
    free_indices = []

    for ind in component:
        if isinstance(ind, Index):
            all_indices.append(ind)
            if ind.count() in fi or ind in free_indices:
                repeated_indices.append(ind)
            free_indices.append(ind)
        elif isinstance(ind, FixedIndex):
            if int(ind) >= shape[len(all_indices)]:
                error("Index out of bounds.")
            all_indices.append(ind)
        elif isinstance(ind, int):
            if int(ind) >= shape[len(all_indices)]:
                error("Index out of bounds.")
            all_indices.append(FixedIndex(ind))
        elif isinstance(ind, slice):
            if ind != slice(None):
                error("Only full slices (:) allowed.")
            i = Index()
            slice_indices.append(i)
            all_indices.append(i)
        elif ind == Ellipsis:
            er = len(shape) - len(component) + 1
            ii = indices(er)
            slice_indices.extend(ii)
            all_indices.extend(ii)
        else:
            error("Not expecting {0}.".format(ind))

    if len(all_indices) != len(shape):
        error("Component and shape length don't match.")

    return tuple(all_indices), tuple(slice_indices), tuple(repeated_indices)


# Outer etc.
def merge_nonoverlapping_indices(a, b):
    """Merge non-overlapping free indices into one representation.

    Example:
      C[i,j,r,s] = outer(A[i,s], B[j,r])
      A, B -> (i,j,r,s), (idim,jdim,rdim,sdim)
    """

    # Extract input properties
    ai = a.ufl_free_indices
    bi = b.ufl_free_indices
    aid = a.ufl_index_dimensions
    bid = b.ufl_index_dimensions

    # Merge lists to return
    s = sorted(zip(ai + bi, aid + bid))
    if s:
        free_indices, index_dimensions = zip(*s)
        # Consistency checks
        if len(set(free_indices)) != len(free_indices):
            error("Not expecting repeated indices.")
    else:
        free_indices, index_dimensions = (), ()
    return free_indices, index_dimensions


# Product
def merge_overlapping_indices(afi, afid, bfi, bfid):
    """Merge overlapping free indices into one free and one repeated representation.

    Example:
      C[j,r] := A[i,j,k] * B[i,r,k]
      A, B -> (j,r), (jdim,rdim), (i,k), (idim,kdim)
    """

    # Extract input properties
    an = len(afi)
    bn = len(bfi)

    # Lists to return
    free_indices = []
    index_dimensions = []
    repeated_indices = []
    repeated_index_dimensions = []

    # Find repeated indices, brute force version
    for i0 in range(an):
        for i1 in range(bn):
            if afi[i0] == bfi[i1]:
                repeated_indices.append(afi[i0])
                repeated_index_dimensions.append(afid[i0])
                break

    # Collect only non-repeated indices, brute force version
    for i, d in sorted(zip(afi + bfi, afid + bfid)):
        if i not in repeated_indices:
            free_indices.append(i)
            index_dimensions.append(d)

    # Consistency checks
    if len(set(free_indices)) != len(free_indices):
        error("Not expecting repeated indices left.")
    if len(free_indices) + 2*len(repeated_indices) != an + bn:
        error("Expecting only twice repeated indices.")

    return tuple(free_indices), tuple(index_dimensions), tuple(repeated_indices), tuple(repeated_index_dimensions)
