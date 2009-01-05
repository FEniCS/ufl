"""This module contains a sorting rule for expr objects that 
is more robust w.r.t. argument numbering than using repr."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-26 -- 2009-01-05"

from ufl.output import ufl_assert
from ufl.common import Counted
from ufl.terminal import Terminal
from ufl.indexing import MultiIndex
from ufl.variable import Label
from ufl.function import Function
from ufl.basisfunction import BasisFunction

#--- Sorting rule ---

def cmp_expr(a, b):
    "Sorting rule for Expr objects."
    # First sort by type name
    aname, bname = a._uflid.__name__, b._uflid.__name__
    c = cmp(aname, bname)
    if c != 0:
        return c
    
    # Type is the same, is it a ...
    # ... MultiIndex object?
    if isinstance(a, MultiIndex):
        # Don't compare counts! Causes circular problems when renumbering to get a canonical form.
        return cmp(len(a), len(b))
    
    # ... Label object?
    if isinstance(a, Label):
        # Don't compare counts! Causes circular problems when renumbering to get a canonical form.
        return 0 # Not equal in general (__eq__ won't be True), but for this purpose they are.
    
    # ... Counted object? (Function or BasisFunction)
    if isinstance(a, Counted):
        ufl_assert(isinstance(a, (Function, BasisFunction)),
            "Expecting a Function or BasisFunction here, got %s instead. Please tell at ufl-dev@fenics.org." % str(type(a)))
        # It's ok to compare counts for form arguments, since their order is a property of the form
        return cmp(a.count(), b.count())
    
    # ... another kind of Terminal object?
    elif isinstance(a, Terminal):
        c = cmp(repr(a), repr(b))
    
    # Not a terminal, sort by number of children (usually the same)
    aops = a.operands()
    bops = b.operands()
    c = cmp(len(aops), len(bops))
    if c != 0:
        return c
    
    # Sort by children in natural order
    for (r,s) in zip(aops, bops):
        c = cmp_expr(r, s)
        if c != 0:
            return c
    
    # All children compare as equal, a and b must be equal
    return 0

