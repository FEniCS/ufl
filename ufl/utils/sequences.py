# -*- coding: utf-8 -*-
"""Various sequence manipulation utilities."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
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

from functools import reduce
import numpy


def product(sequence):
    """Return the product of all elements in a sequence."""
    p = 1
    for f in sequence:
        p *= f
    return p


def unzip(seq):
    """Inverse operation of zip:

    unzip(zip(a, b)) == (a, b)."""
    return [s[0] for s in seq], [s[1] for s in seq]


def xor(a, b):
    return bool(a) if b else not a


def or_tuples(seqa, seqb):
    """Return 'or' of all pairs in two sequences of same length."""
    return tuple(a or b for (a, b) in zip(seqa, seqb))


def and_tuples(seqa, seqb):
    """Return 'and' of all pairs in two sequences of same length."""
    return tuple(a and b for (a, b) in zip(seqa, seqb))


def iter_tree(tree):
    """Iterate over all nodes in a tree represented
    by lists of lists of leaves."""
    if isinstance(tree, list):
        for node in tree:
            for i in iter_tree(node):
                yield i
    else:
        yield tree


def recursive_chain(lists):
    for l in lists:
        if isinstance(l, str):
            yield l
        else:
            for s in recursive_chain(l):
                yield s


def max_degree(degrees):
    """Maximum degree for mixture of scalar and tuple degrees."""
    # numpy.maximum broadcasts scalar degrees to tuple degrees if
    # necessary.  reduce applies numpy.maximum pairwise.
    degree = reduce(numpy.maximum, map(numpy.asarray, degrees))
    if degree.ndim:
        degree = tuple(map(int, degree))  # tuple degree
    else:
        degree = int(degree)              # scalar degree
    return degree
