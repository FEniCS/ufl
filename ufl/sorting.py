"""Sorting.

This module contains a sorting rule for expr objects that
is more robust w.r.t. argument numbering than using repr.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010.
# Modified by Johan Hake, 2010.

from functools import cmp_to_key

from ufl.core.expr import Expr
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.core.multiindex import FixedIndex, MultiIndex
from ufl.variable import Label


def _cmp_multi_index(a, b):
    """Cmp multi index."""
    # Careful not to depend on Index.count() here!
    # This is placed first because it is most frequent.
    # Make decision based on the first index pair possible
    for i, j in zip(a._indices, b._indices):
        fix1 = isinstance(i, FixedIndex)
        fix2 = isinstance(j, FixedIndex)
        if fix1 and fix2:
            # Both are FixedIndex, sort by value
            x, y = i._value, j._value
            if x < y:
                return -1
            elif x > y:
                return 1
            else:
                # Same value, no decision
                continue
        elif fix1:
            # Sort fixed before free
            return -1
        elif fix2:
            # Sort fixed before free
            return 1
        else:
            # Both are Index, no decision, do not depend on count!
            pass
    # Failed to make a decision, return 0 by default
    # (this does not mean equality, it could be e.g.
    # [i,0] vs [j,0] because the counts of i,j cannot be used)
    return 0


def _cmp_label(a, b):
    """Cmp label."""
    # Don't compare counts! Causes circular problems when renumbering to get a canonical form.
    # Therefore, even though a and b are not equal in general (__eq__ won't be True),
    # but for this sorting they are considered equal and we return 0.
    return 0


def _cmp_coefficient(a, b):
    """Cmp coefficient."""
    # It's ok to compare relative counts for Coefficients,
    # since their ordering is a property of the form
    x, y = a._count, b._count
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0


def _cmp_argument(a, b):
    """Cmp argument."""
    # It's ok to compare relative number and part for Arguments,
    # since their ordering is a property of the form
    x = (a._number, a._part)
    y = (b._number, b._part)
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0


def _cmp_terminal_by_repr(a, b):
    """Cmp terminal by repr."""
    # The cost of repr on a terminal is fairly small, and bounded
    x = repr(a)
    y = repr(b)
    return -1 if x < y else (0 if x == y else 1)


# Hack up a MultiFunction-like type dispatch for terminal comparisons
_terminal_cmps = [_cmp_terminal_by_repr] * Expr._ufl_num_typecodes_
_terminal_cmps[MultiIndex._ufl_typecode_] = _cmp_multi_index
_terminal_cmps[Argument._ufl_typecode_] = _cmp_argument
_terminal_cmps[Coefficient._ufl_typecode_] = _cmp_coefficient
_terminal_cmps[Label._ufl_typecode_] = _cmp_label


def cmp_expr(a, b):
    """Replacement for cmp(a, b), removed in Python 3, for Expr objects."""
    # Modelled after pre_traversal to avoid recursion:
    left = [(a, b)]
    while left:
        a, b = left.pop()

        # First sort quickly by type code
        x, y = a._ufl_typecode_, b._ufl_typecode_
        if x != y:
            return -1 if x < y else 1

        # Now we know that the type is the same, check further based
        # on type specific properties.
        if a._ufl_is_terminal_:
            c = _terminal_cmps[x](a, b)
            if c:
                return c
        else:
            # If the hash is the same, assume equal for the purpose of
            # sorting.  This introduces a minor chance of
            # nondeterministic behaviour, just as with MultiIndex.
            # Although collected statistics for complicated forms
            # suggest that the hash function is pretty good so there
            # shouldn't be collisions.
            # if hash(a) == hash(b): # FIXME: Test this for performance improvement.
            #    return 0

            # Delve into subtrees
            aops = a.ufl_operands
            bops = b.ufl_operands

            # Sort by children in natural order
            for (r, s) in zip(aops, bops):
                # Skip subtree if objects are the same
                if r is s:
                    continue
                # Append subtree for further inspection
                left.append((r, s))

            # All children compare as equal, a and b must be
            # equal. Except for...  A few types, notably ExprList and
            # ExprMapping, can have a different number of operands.
            # Sort by the length if it's different. Doing this after
            # sorting by children because these types are rare so we
            # try to avoid the cost of this check for most nodes.
            x, y = len(aops), len(bops)
            if x != y:
                return -1 if x < y else 1

    # Equal if we get out of the above loop!
    return 0


def sorted_expr(sequence):
    """Return a canonically sorted list of Expr objects in sequence."""
    return sorted(sequence, key=cmp_to_key(cmp_expr))


def sorted_expr_sum(seq):
    """Sorted expr sum."""
    seq2 = sorted(seq, key=cmp_to_key(cmp_expr))
    s = seq2[0]
    for e in seq2[1:]:
        s = s + e
    return s
