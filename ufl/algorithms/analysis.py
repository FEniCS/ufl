"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-29"

# Modified by Anders Logg, 2008

from itertools import chain

from ..output import ufl_assert
from ..common import lstr
from ..base import UFLObject
from ..algebra import Sum, Product
from ..basisfunction import BasisFunction
from ..function import Function
from ..indexing import DefaultDimType
from ..form import Form
from ..integral import Integral
from .traversal import iter_expressions, post_traversal

#--- Utilities to extract information from an expression ---

def extract_type(a, ufl_type):
    """Build a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or UFLObject."""
    iter = (o for e in iter_expressions(a) \
              for (o, stack) in post_traversal(e) \
              if isinstance(o, ufl_type) )
    return set(iter)

def classes(a):
    """Build a set of all unique UFLObject subclasses used in a.
    The argument a can be a Form, Integral or UFLObject."""
    c = set()
    for e in iter_expressions(a):
        for (o, stack) in post_traversal(e):
            c.add(o.__class__)
    return c

def domain(a):
    "Find the polygonal domain of Form a."
    el = elements(a)
    dom = el[0].domain()
    return dom

def value_shape(expression, dimension):
    "Evaluate the value shape of expression with given implicit dimension."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFL expression.")
    ufl_assert(isinstance(dimension, int), "Expecting int dimension.")
    s = expression.shape()
    shape = []
    for i in s:
        if isinstance(i, DefaultDimType):
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
        indices.update(i for i in mi if isinstance(i, Index))
    return indices

def duplications(expression):
    "Build a set of all repeated expressions in expression."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFL expression.")
    handled = set()
    duplicated = set()
    for (o, stack) in post_traversal(expression):
        if o in handled:
            duplicated.add(o)
        handled.add(o)
    return duplicated

def monomials(expression):
    "Compute monomial representation of expression (if possible)."

    # FIXME: Not yet working, need to include derivatives, integrals etc

    ufl_assert(isinstance(expression, Form) or isinstance(expression, UFLObject), "Expecting UFL form or expression.")

    # Iterate over expressions
    m = []
    for e in iter_expressions(expression):
        operands = e.operands()
        if isinstance(e, Sum):
            ufl_assert(len(operands) == 2, "Strange, expecting two terms.")
            m += monomials(operands[0])
            m += monomials(operands[1])
        elif isinstance(e, Product):
            ufl_assert(len(operands) == 2, "Strange, expecting two factors.")
            for m0 in monomials(operands[0]):
                for m1 in monomials(operands[1]):
                    m.append(m0 + m1)
        elif isinstance(e, BasisFunction):
            m.append((e,))

    return m
