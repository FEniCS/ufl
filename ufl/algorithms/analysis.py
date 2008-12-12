"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-12-12"

# Modified by Anders Logg, 2008

from itertools import chain

from ufl.output import ufl_assert, ufl_error, ufl_info
from ufl.common import lstr, UFLTypeDefaultDict

from ufl.base import Expr, Terminal
from ufl.algebra import Sum, Product, Division
from ufl.finiteelement import MixedElement
from ufl.basisfunction import BasisFunction
from ufl.function import Function
from ufl.variable import Variable
from ufl.function import Function, Constant
from ufl.tensors import ListTensor, ComponentTensor
from ufl.tensoralgebra import Transposed, Inner, Dot, Outer, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor, Skew
from ufl.restriction import PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ufl.indexing import Indexed, Index, MultiIndex
from ufl.form import Form
from ufl.integral import Integral
from ufl.classes import terminal_classes, nonterminal_classes
from ufl.algorithms.traversal import iter_expressions, post_traversal, walk


#--- Utilities to extract information from an expression ---

def extract_classes(a):
    """Build a set of all unique Expr subclasses used in a.
    The argument a can be a Form, Integral or Expr."""
    return set(o._uflid for e in iter_expressions(a) \
                        for (o, stack) in post_traversal(e))

def extract_type(a, ufl_type):
    """Build a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or Expr."""
    return set(o for e in iter_expressions(a) \
                 for (o, stack) in post_traversal(e) \
                 if isinstance(o, ufl_type))

def extract_terminals(a):
    "Build a set of all Terminal objects in a."
    return set(o for e in iter_expressions(a) \
                 for (o,stack) in post_traversal(e) \
                 if isinstance(o, Terminal))

def extract_basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or Expr."""
    def c(x,y):
        return cmp(x._count, y._count)
    return sorted(extract_type(a, BasisFunction), cmp=c)

def extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    def c(x,y):
        return cmp(x._count, y._count)
    return sorted(extract_type(a, Function), cmp=c)

# alternative implementation, kept as an example:
def _extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Function):
            s.add(o)
    walk(a, func)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x.count(), y.count()))
    return l

def extract_elements(a):
    "Build a sorted list of all elements used in a."
    args = chain(extract_basisfunctions(a), extract_coefficients(a))
    return tuple(f.element() for f in args)

def extract_unique_elements(a):
    "Build a set of all unique elements used in a."
    return set(extract_elements(a))

def extract_sub_elements(element):
    "Build a set of all unique subelements, including parent element." 
    res = set((element,))
    if isinstance(element, MixedElement):
        for sub in element.sub_elements():
            res.update(extract_sub_elements(sub))
    return res

def extract_indices(expression):
    "Build a set of all Index objects used in expression."
    ufl_info("Is this used for anything? Doesn't make much sense.")
    multi_indices = extract_type(expression, MultiIndex)
    indices = set()
    for mi in multi_indices:
        indices.update(i for i in mi if isinstance(i, Index))
    return indices

def extract_variables(a):
    """Build a set of all Variable objects in a,
    which can be a Form, Integral or Expr."""
    return extract_type(a, Variable)
# FIXME: Does these two do the exact same thing?
def extract_variables(expression, handled_vars=None):
    if handled_vars is None:
        handled_vars = set()
    if isinstance(expression, Variable):
        i = expression._count
        if i in handled_vars:
            return []
        handled_vars.add(i)
        variables = list(extract_variables(expression._expression, handled_vars))
        variables.append(expression)
    else:
        variables = []
        for o in expression.operands():
            variables.extend(extract_variables(o, handled_vars))
    return variables

def extract_duplications(expression):
    "Build a set of all repeated expressions in expression."
    # TODO: Handle indices in a canonical way, maybe create a transformation that does this to apply before extract_duplications?
    ufl_assert(isinstance(expression, Expr), "Expecting UFL expression.")
    handled = set()
    duplicated = set()
    for (o, stack) in post_traversal(expression):
        if o in handled:
            duplicated.add(o)
        handled.add(o)
    return duplicated

