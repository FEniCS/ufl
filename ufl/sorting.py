"""This module contains a sorting rule for expr objects that
is more robust w.r.t. argument numbering than using repr."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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

from itertools import izip

from ufl.log import error
from ufl.terminal import Terminal, FormArgument
from ufl.indexing import Index, FixedIndex, MultiIndex
from ufl.variable import Label

def _cmp3(a, b):
    "Replacement for cmp(), removed in Python 3."
    # TODO: Which is faster?
    return -1 if (a < b) else (+1 if a > b else 0)
    #return (a > b) - (a < b)

def cmp_expr(a, b):
    "Sorting rule for Expr objects."
    # First sort quickly by type name
    c = _cmp3(a._uflclass.__name__, b._uflclass.__name__)
    if c != 0:
        return c

    # Type is the same, but is it a ...

    # ... a MultiIndex? Careful not to depend on Index.count() here!
    if isinstance(a, MultiIndex):
        for i,j in izip(a._indices, b._indices):
            if isinstance(i, FixedIndex):
                if isinstance(j, FixedIndex):
                    # Both are FixedIndex, sort by value
                    c = _cmp3(i._value, j._value)
                    if c:
                        return c
                else:
                    return +1
            else:
                if isinstance(j, FixedIndex):
                    return -1
                else:
                    pass # Both are Index, do not depend on count!
        # Failed to make a decision, return 0 by default
        return 0

    # ... Label object?
    elif isinstance(a, Label):
        # Don't compare counts! Causes circular problems when renumbering to get a canonical form.
        # Therefore, even though a and b are not equal in general (__eq__ won't be True),
        # but for this sorting they are considered equal and we return 0.
        return 0

    # ... Other Counted object? (Coefficient or Argument)
    elif isinstance(a, FormArgument):
        # It's ok to compare relative counts for form arguments,
        # since their ordering is a property of the form
        return _cmp3(a._count, b._count)

    # ... another kind of Terminal object?
    elif isinstance(a, Terminal):
        # The cost of repr on a terminal is fairly small, and bounded
        return _cmp3(repr(a), repr(b))

    # Not a terminal, sort by number of children (usually the same)
    aops = a.operands()
    bops = b.operands()
    c = _cmp3(len(aops), len(bops))
    if c != 0:
        return c

    # Sort by children in natural order
    for (r, s) in izip(aops, bops):
        c = cmp_expr(r, s) # Ouch! This becomes worst case O(n) then? FIXME: Perhaps replace with comparison of hash value? Is that stable between runs?
        if c != 0:
            return c

    # All children compare as equal, a and b must be equal
    return 0

# Not in python 2.6...
#from functools import cmp_to_key

class ExprKey(object):
    __slots__ = ('x',)
    def __init__(self, x):
        self.x = x
    def __lt__(self, other):
        return cmp_expr(self.x, other.x) < 0

def expr_key(expr):
    return ExprKey(expr)

def sorted_expr(seq):
    return sorted(seq, key=expr_key)

def sorted_expr_sum(seq):
    seq2 = sorted(seq, key=expr_key)
    s = seq2[0]
    for e in seq2[1:]:
        s = s + e
    return s

# TODO: Move this to common.py, does not belong here
def topological_sorting(nodes, edges):
    """
    Return a topologically sorted list of the nodes

    Implemented algorithm from Wikipedia :P

    <http://en.wikipedia.org/wiki/Topological_sorting>

    No error for cyclic edges...
    """

    L = []
    S = nodes[:]
    for node in nodes:
        for es in edges.itervalues():
            if node in es and node in S:
                S.remove(node)
                continue

    while S:
        node = S.pop(0)
        L.append(node)
        node_edges = edges[node]
        while node_edges:
            m = node_edges.pop(0)
            found = False
            for es in edges.itervalues():
                found = m in es
                if found:
                    break
            if not found:
                S.insert(0,m)

    return L
