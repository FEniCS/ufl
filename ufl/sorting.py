# -*- coding: utf-8 -*-
"""This module contains a sorting rule for expr objects that
is more robust w.r.t. argument numbering than using repr."""

# Copyright (C) 2008-2014 Martin Sandve Alnæs
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
#
# Modified by Anders Logg, 2009-2010.
# Modified by Johan Hake, 2010.
#
# First added:  2008-11-26
# Last changed: 2013-01-02

from six.moves import zip

from ufl.log import error
from ufl.core.terminal import Terminal
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.core.multiindex import Index, FixedIndex, MultiIndex
from ufl.variable import Label
from ufl.geometry import GeometricQuantity


def _cmp3(a, b):
    "Replacement for cmp(), removed in Python 3."
    return -1 if (a < b) else (+1 if a > b else 0)


def _cmp_terminal(a, b):
    # Is it a...
    # ... MultiIndex? Careful not to depend on Index.count() here! This is placed first because it is most frequent.
    if isinstance(a, MultiIndex):
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
                    return +1
                else:
                    # Same value, no decision
                    continue
            elif fix1:
                # Sort fixed before free
                return -1
            elif fix2:
                # Sort fixed before free
                return +1
            else:
                # Both are Index, no decision, do not depend on count!
                pass
        # Failed to make a decision, return 0 by default
        # (this does not mean equality, it could be e.g.
        # [i,0] vs [j,0] because the counts of i,j cannot be used)
        return 0

    # ... Label object?
    elif isinstance(a, Label):
        # Don't compare counts! Causes circular problems when renumbering to get a canonical form.
        # Therefore, even though a and b are not equal in general (__eq__ won't be True),
        # but for this sorting they are considered equal and we return 0.
        return 0

    # ... Coefficient?
    elif isinstance(a, Coefficient):
        # It's ok to compare relative counts for Coefficients,
        # since their ordering is a property of the form
        x, y = a._count, b._count
        if x < y:
            return -1
        elif x > y:
            return +1
        else:
            return 0

    # ... Argument?
    elif isinstance(a, Argument):
        # It's ok to compare relative number and part for Arguments,
        # since their ordering is a property of the form
        x = (a._number, a._part)
        y = (b._number, b._part)
        if x < y:
            return -1
        elif x > y:
            return +1
        else:
            return 0

    # ... another kind of Terminal object?
    else:
        # The cost of repr on a terminal is fairly small, and bounded
        x, y = repr(a), repr(b)
        if x < y:
            return -1
        elif x > y:
            return +1
        else:
            return 0


def _cmp_operator(a, b):
    # If the hash is the same, assume equal for the purpose of sorting.
    # This introduces a minor chance of nondeterministic behaviour, just as with MultiIndex.
    # Although collected statistics for complicated forms suggest that the hash
    # function is pretty good so there shouldn't be collisions.
    if hash(a) == hash(b): # FIXME: Test this for performance improvement.
        return 0

    aops = a.ufl_operands
    bops = b.ufl_operands

    # Sort by children in natural order
    for (r, s) in zip(aops, bops):
        # Ouch! This becomes worst case O(n) then?
        # FIXME: Perhaps replace with comparison of hash value? But that's not stable between python versions.
        c = cmp_expr(r, s)
        if c:
            return c

    # All children compare as equal, a and b must be equal. Except for...
    # A few type, notably ExprList and ExprMapping, can have a different number of operands.
    # Sort by the length if it's different. Doing this after sorting by children because
    # these types are rare so we try to avoid the cost of this check for most nodes.
    return _cmp3(len(aops), len(bops))


def cmp_expr2(a, b):
    "Sorting rule for Expr objects. NB! Do not use to compare for equality!"

    # First sort quickly by type code
    c = _cmp3(a._ufl_typecode_, b._ufl_typecode_)
    if c:
        return c

    # Now we know that the type is the same, check further based on type specific properties.
    if a._ufl_is_terminal_:
        return _cmp_terminal(a, b)
    else:
        return _cmp_operator(a, b)


# FIXME: Test and benchmark this! Could be faster since it avoids the recursion.
def cmp_expr(a, b):

    # Modelled after pre_traversal to avoid recursion:
    left = [(a, b)]
    while left:
        a, b = left.pop()

        # First sort quickly by type code
        x, y = a._ufl_typecode_, b._ufl_typecode_
        if x < y:
            return -1
        elif x > y:
            return +1

        # Now we know that the type is the same, check further based on type specific properties.
        if a._ufl_is_terminal_:
            c = _cmp_terminal(a, b)
            if c:
                return c
        else:
            # If the hash is the same, assume equal for the purpose of sorting.
            # This introduces a minor chance of nondeterministic behaviour, just as with MultiIndex.
            # Although collected statistics for complicated forms suggest that the hash
            # function is pretty good so there shouldn't be collisions.
            #if hash(a) == hash(b): # FIXME: Test this for performance improvement.
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

            # All children compare as equal, a and b must be equal. Except for...
            # A few type, notably ExprList and ExprMapping, can have a different number of operands.
            # Sort by the length if it's different. Doing this after sorting by children because
            # these types are rare so we try to avoid the cost of this check for most nodes.
            x, y = len(aops), len(bops)
            if x < y:
                return -1
            elif x > y:
                return +1

    # Equal if we get out of the above loop!
    return 0



# Not in python 2.6...
#from functools import cmp_to_key


class ExprKey(object):
    __slots__ = ('x',)

    def __init__(self, x):
        self.x = x

    def __lt__(self, other):
        return cmp_expr(self.x, other.x) < 0


def sorted_expr(seq):
    return sorted(seq, key=ExprKey)


def sorted_expr_sum(seq):
    seq2 = sorted(seq, key=ExprKey)
    s = seq2[0]
    for e in seq2[1:]:
        s = s + e
    return s


from ufl.common import topological_sorting # FIXME: Remove this, update whoever uses it in ufl and ffc etc.
