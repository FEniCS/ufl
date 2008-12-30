"""This module contains a sorting rule for expr objects that 
is more robust w.r.t. argument numbering than using repr."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-26 -- 2008-11-26"

from ufl.common import Counted
from ufl.terminal import Terminal

#--- Sorting rule ---

def cmp_expr(a, b):
    "Sorting rule for Expr objects."
    # First sort by type name
    aname, bname = a._uflid.__name__, b._uflid.__name__
    c = cmp(aname, bname)
    if c != 0:
        return c
    # Type is the same, is it a Counted object?
    if isinstance(a, Counted):
        return cmp(a.count(), b.count())
    # Is it another kind of Terminal object?
    elif isinstance(a, Terminal):
        c = cmp(repr(a), repr(b))
    # Not a terminal, sort by number of children (usually the same)
    aops = a.operands()
    bops = b.operands()
    c = cmp(len(aops), len(bops))
    if c != 0:
        return c
    # Sort by order of children
    for (r,s) in zip(aops, bops):
        c = cmp_expr(r, s)
        if c != 0:
            return c
    # All children compare as equal, a and b must be equal
    return 0

