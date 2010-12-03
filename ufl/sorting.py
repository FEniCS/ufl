"""This module contains a sorting rule for expr objects that
is more robust w.r.t. argument numbering than using repr."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-11-26 -- 2009-04-25"

# Modified by Anders Logg, 2009-2010.
# Modified by Johan Hake, 2010.
# Last changed: 2010-01-26

from itertools import izip

from ufl.log import error
from ufl.common import Counted
from ufl.terminal import Terminal, FormArgument
from ufl.indexing import MultiIndex
from ufl.variable import Label
from ufl.argument import Argument
from ufl.coefficient import Coefficient

def cmp_expr(a, b):
    "Sorting rule for Expr objects."
    # First sort quickly by type name
    c = cmp(a._uflclass.__name__, b._uflclass.__name__)
    if c != 0:
        return c

    # Type is the same, but is it a ...

    # ... Label object?
    if isinstance(a, Label):
        # Don't compare counts! Causes circular problems when renumbering to get a canonical form.
        return 0 # Not equal in general (__eq__ won't be True), but for this purpose they are considered equal.

    # ... Other Counted object? (Function or Argument)
    elif isinstance(a, Counted):
        if not isinstance(a, FormArgument):
            error("Expecting a Function or Argument here, got %s instead. Please tell at ufl-dev@fenics.org." % str(type(a)))
        # It's ok to compare counts for form arguments, since their order is a property of the form
        return cmp(a._count, b._count)

    # ... another kind of Terminal object?
    elif isinstance(a, Terminal) and not isinstance(a, MultiIndex):

        c = cmp(repr(a), repr(b))

    # Not a terminal, sort by number of children (usually the same)
    aops = a.operands()
    bops = b.operands()
    c = cmp(len(aops), len(bops))
    if c != 0:
        return c

    # Sort by children in natural order
    for (r, s) in izip(aops, bops):
        c = cmp_expr(r, s)
        if c != 0:
            return c

    # All children compare as equal, a and b must be equal
    return 0

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
