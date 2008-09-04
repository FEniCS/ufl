"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-04"

from itertools import chain

from ..output import ufl_assert
from ..base import UFLObject
from ..basisfunctions import BasisFunction, Function
from ..indexing import UnassignedDimType
from ..classes import Form, Integral
from .traversal import iter_expressions, post_traversal

#--- Utilities to extract information from an expression ---

def extract_type(a, ufl_type):
    """Build a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or UFLObject."""
    iter = (o for e in iter_expressions(a) \
              for o in post_traversal(e) \
              if isinstance(o, ufl_type) )
    return set(iter)

def classes(a):
    """Build a set of all unique UFLObject subclasses used in a.
    The argument a can be a Form, Integral or UFLObject."""
    classes = set()
    for e in iter_expressions(a):
        for o in post_traversal(e):
            classes.add(o.__class__)
    return classes

def domain(a):
    "Find the polygonal domain of Form a."
    el = elements(a)
    dom = el[0].domain()
    return dom

def value_shape(expression, dimension):
    "Evaluate the value shape of expression with given implicit dimension."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject expression.")
    ufl_assert(isinstance(dimension, int), "Expecting int dimension.")
    s = expression.shape()
    shape = []
    for i in s:
        if isinstance(i, UnassignedDimType):
            shape.append(dimension)
        else:
            shape.append(i)
    return tuple(shape)

def basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or UFLObject."""
    # build set of all unique basisfunctions
    s = extract_type(a, BasisFunction)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject."""
    # build set of all unique coefficients
    s = extract_type(a, Function)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

# alternative implementation, kept as an example:
def _coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject."""
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
    "Build a sorted list of all elements used in a."
    return [f._element for f in chain(basisfunctions(a), coefficients(a))]

def unique_elements(a):
    "Build a set of all unique elements used in a."
    return set(elements(a))

def variables(a):
    """Build a set of all Variable objects in a,
    which can be a Form, Integral or UFLObject."""
    return extract_type(a, Variable)

def indices(expression):
    "Build a set of all Index objects used in expression."
    multi_indices = extract_type(expression, MultiIndex)
    indices = set()
    for mi in multi_indices:
        indices.update(i for i in mi._indices if isinstance(i, Index))
    return indices

def duplications(expression):
    "Build a set of all repeated expressions in expression."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject.")
    handled = set()
    duplicated = set()
    for o in post_traversal(expression):
        if o in handled:
            duplicated.add(o)
        handled.add(o)
    return duplicated

class FormData(object):
    "Class collecting various information extracted from form."
    def __init__(self, form):
        ufl_assert(isinstance(form, Form), "Expecting Form.")
        self.basisfunctions  = basisfunctions(form)
        self.coefficients    = coefficients(form)
        self.elements        = elements(form)
        self.unique_elements = unique_elements(form)
        self.domain          = domain(form)
        self.classes         = classes(form)

