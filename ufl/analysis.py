"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"

from itertools import chain

from base import *
from traversal import *


#--- Utilities to extract information from an expression ---

def extract_type(a, ufl_type):
    """Returns a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or UFLObject.
    """
    iter = (o for e in iter_expressions(a) \
              for o in post_traversal(e) \
              if isinstance(o, ufl_type) )
    return set(iter)

def basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique basisfunctions
    s = extract_type(a, BasisFunction)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique coefficients
    s = extract_type(a, Function)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

# alternative implementation, kept as an example:
def _coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Function):
            s.add(o)
    walk(a, func)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def elements(a):
    """Returns a sorted list of all elements used in a."""
    return [f._element for f in chain(basisfunctions(a), coefficients(a))]

def unique_elements(a):
    """Returns a set of all elements used in a."""
    return set(elements(a))

def classes(a):
    """Returns a set of all unique UFLObject subclasses used in a."""
    classes = set()
    for e in iter_expressions(a):
        for o in post_traversal(e):
            classes.add(o.__class__)
    return classes

def variables(a):
    """Returns a set of all variables in a,
    which can be a Form, Integral or UFLObject.
    """
    return extract_type(a, Variable)

def duplications(a):
    """Returns a set of all repeated expressions in u."""
    handled = set()
    duplicated = set()
    for e in iter_expressions(a):
        for o in post_traversal(e):
            if o in handled:
                duplicated.add(o)
            handled.add(o)
    return duplicated

