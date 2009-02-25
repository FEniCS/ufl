"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-25"

# Modified by Anders Logg, 2008

from itertools import chain

from ufl.log import error, info
from ufl.assertions import ufl_assert
from ufl.common import lstr, dstr, UFLTypeDefaultDict

from ufl.expr import Expr
from ufl.terminal import Terminal, FormArgument
from ufl.algebra import Sum, Product, Division
from ufl.finiteelement import MixedElement
from ufl.basisfunction import BasisFunction
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
from ufl.algorithms.traversal import iter_expressions, post_traversal, post_walk, traverse_terminals

#--- Utilities to extract information from an expression ---

def extract_classes(a):
    """Build a set of all unique Expr subclasses used in a.
    The argument a can be a Form, Integral or Expr."""
    return set(o._uflclass for e in iter_expressions(a) \
                        for o in post_traversal(e))

def extract_type(a, ufl_type):
    """Build a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or Expr."""
    if issubclass(ufl_type, Terminal):
        return set(o for e in iter_expressions(a) \
                     for o in traverse_terminals(e) \
                     if isinstance(o, ufl_type))
    return set(o for e in iter_expressions(a) \
                 for o in post_traversal(e) \
                 if isinstance(o, ufl_type))

def extract_terminals(a):
    "Build a set of all Terminal objects in a."
    return set(o for e in iter_expressions(a) \
                 for o in post_traversal(e) \
                 if isinstance(o, Terminal))

def cmp_counted(x, y):
    return cmp(x._count, y._count)

def extract_basis_functions(a):
    """Build a sorted list of all basis_functions in a,
    which can be a Form, Integral or Expr."""
    return sorted(extract_type(a, BasisFunction), cmp=cmp_counted)

def extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    return sorted(extract_type(a, Function), cmp=cmp_counted)

def extract_arguments(a):
    """Build two sorted lists of all basis functions and functions 
    in a, which can be a Form, Integral or Expr."""
    terminals = extract_type(a, FormArgument)
    basis_functions = [f for f in terminals if isinstance(f, BasisFunction)]
    functions = [f for f in terminals if isinstance(f, Function)]
    basis_functions = sorted(basis_functions, cmp=cmp_counted)
    functions = sorted(functions, cmp=cmp_counted)
    return basis_functions, functions

def build_argument_replace_map(basis_functions, functions):
    """Create new BasisFunction and Function objects
    with count starting at 0. Return mapping from old
    to new objects, and lists of the new objects."""
    new_basis_functions = [f.reconstruct(count=i)\
                           for (i, f) in enumerate(basis_functions)]
    new_functions       = [f.reconstruct(count=i)\
                           for (i, f) in enumerate(functions)]
    replace_map = {}
    replace_map.update(zip(basis_functions, new_basis_functions))
    replace_map.update(zip(functions, new_functions))
    return replace_map, new_basis_functions, new_functions

# alternative implementation, kept as an example:
def _extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Function):
            s.add(o)
    post_walk(a, func)
    # sort by count
    return sorted(s, cmp=cmp_counted)

def extract_elements(a):
    "Build a sorted list of all elements used in a."
    args = chain(extract_basis_functions(a), extract_coefficients(a))
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
    info("Is this used for anything? Doesn't make much sense.")
    multi_indices = extract_type(expression, MultiIndex)
    indices = set()
    for mi in multi_indices:
        indices.update(i for i in mi if isinstance(i, Index))
    return indices

def extract_variables(a):
    """Build a list of all Variable objects in a,
    which can be a Form, Integral or Expr.
    The ordering in the list obeys dependency order."""
    handled = set()
    variables = []
    for e in iter_expressions(a):
        for o in post_traversal(e):
            if isinstance(o, Variable):
                expr, label = o.operands()
                if not label in handled:
                    variables.append(o)
                    handled.add(label)
    return variables    

def extract_duplications(expression):
    "Build a set of all repeated expressions in expression."
    # TODO: Handle indices in a canonical way, maybe create a transformation that does this to apply before extract_duplications?
    ufl_assert(isinstance(expression, Expr), "Expecting UFL expression.")
    handled = set()
    duplicated = set()
    for o in post_traversal(expression):
        if o in handled:
            duplicated.add(o)
        handled.add(o)
    return duplicated

def count_nodes(expr, ids=None):
    "Count the number of unique Expr instances in expression."
    i = id(expr)
    if ids is None:
        ids = set()
    elif i in ids:
        # Skip already visited subtrees
        return
    # Extend set with children recursively
    for o in expr.operands():
        count_nodes(o, ids)
    ids.add(i)
    return len(ids)

def extract_quadrature_order(integral):
    """Extract quadrature integration order from quadrature
    elements in integral. Returns None if not found."""
    quadrature_elements = [e for e in extract_elements(integral) if "Quadrature" in e.family()]
    if not quadrature_elements:
        return None
    quad_order = quadrature_elements[0].degree()
    ufl_assert(all(quad_order == e.degree() for e in quadrature_elements),
        "Incompatible quadrature elements specified (orders must be equal).")
    return quad_order

def estimate_quadrature_order(integral):
    "Estimate the necessary quadrature order for integral using the sum of basis function degrees."
    return sum(b.element().degree() for b in extract_basis_functions(integral))

def estimate_max_quadrature_order(integral):
    "Estimate the maximum needed quadrature order for integral using the highest polynomial degree of any term."
    # TODO: Provide a different estimate as an additional algorithm, finding the actual max polynomial order including order of functions.
    return estimate_quadrature_order(integral)

