"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-03-14 -- 2009-04-17"

# Modified by Anders Logg, 2009-2010.
# Modified by Johan Hake, 2010.
# Last changed: 2010-01-26

from itertools import chain

from ufl.log import error, warning, info
from ufl.assertions import ufl_assert
from ufl.common import lstr, dstr, UFLTypeDefaultDict
from ufl.sorting import topological_sorting

from ufl.expr import Expr
from ufl.terminal import Terminal, FormArgument
from ufl.algebra import Sum, Product, Division
from ufl.finiteelement import MixedElement
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.variable import Variable
from ufl.tensors import ListTensor, ComponentTensor
from ufl.tensoralgebra import Transposed, Inner, Dot, Outer, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor, Skew
from ufl.restriction import PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ufl.indexing import Indexed, Index, MultiIndex
from ufl.form import Form
from ufl.integral import Integral
from ufl.classes import terminal_classes, nonterminal_classes
from ufl.algorithms.traversal import iter_expressions, post_traversal, post_walk, traverse_terminals

# Domain types (should probably be listed somewhere else)
_domain_types = ("cell", "exterior_facet", "interior_facet", "macro_cell", "surface")

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

def has_type(a, ufl_types):
    """Check if any class from ufl_types is found in a.
    The argument a can be a Form, Integral or Expr."""
    if issubclass(ufl_types, Expr):
        ufl_types = (ufl_types,)
    if all(issubclass(ufl_type, Terminal) for ufl_type in ufl_types):
        return any(isinstance(o, ufl_types) \
                   for e in iter_expressions(a) \
                   for o in traverse_terminals(e))
    return any(isinstance(o, ufl_types) \
               for e in iter_expressions(a) \
               for o in post_traversal(e))

def extract_terminals(a):
    "Build a set of all Terminal objects in a."
    return set(o for e in iter_expressions(a) \
                 for o in post_traversal(e) \
                 if isinstance(o, Terminal))

def cmp_counted(x, y):
    return cmp(x._count, y._count)

def extract_arguments(a):
    """Build a sorted list of all arguments in a,
    which can be a Form, Integral or Expr."""
    return sorted(extract_type(a, Argument), cmp=cmp_counted)

def extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    return sorted(extract_type(a, Coefficient), cmp=cmp_counted)

# FIXME: Why is this extra function needed, why not use
# FIXME: the two functions above?

def extract_arguments_and_coefficients(a):
    """Build two sorted lists of all arguments and coefficients
    in a, which can be a Form, Integral or Expr."""

    # Extract lists of all form argument instances
    terminals = extract_type(a, FormArgument)
    arguments = [f for f in terminals if isinstance(f, Argument)]
    coefficients = [f for f in terminals if isinstance(f, Coefficient)]

    # Build count: instance mappings, should be one to one
    bfcounts = dict((f, f.count()) for f in arguments)
    fcounts = dict((f, f.count()) for f in coefficients)

    if len(bfcounts) != len(set(bfcounts.values())):
        msg = """\
Found different basis function arguments with same counts.
Did you combine test or trial functions from different spaces?
The arguments found are:\n%s""" % "\n".join("  %s" % f for f in arguments)
        error(msg)

    if len(fcounts) != len(set(fcounts.values())):
        msg = """\
Found different functions with same counts.
The arguments found are:\n%s""" % "\n".join("  %s" % f for f in coefficients)
        error(msg)

    # Passed checks, so we can safely sort the instances by count
    arguments = sorted(arguments, cmp=cmp_counted)
    coefficients = sorted(coefficients, cmp=cmp_counted)

    return arguments, coefficients

def build_argument_replace_map(arguments, coefficients):
    """Create new Argument and Coefficient objects
    with count starting at 0. Return mapping from old
    to new objects, and lists of the new objects."""
    new_arguments = [f.reconstruct(count=i)\
                           for (i, f) in enumerate(arguments)]
    new_coefficients       = [f.reconstruct(count=i)\
                           for (i, f) in enumerate(coefficients)]
    replace_map = {}
    replace_map.update(zip(arguments, new_arguments))
    replace_map.update(zip(coefficients, new_coefficients))
    return replace_map, new_arguments, new_coefficients

# alternative implementation, kept as an example:
def _extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Coefficient):
            s.add(o)
    post_walk(a, func)
    # sort by count
    return sorted(s, cmp=cmp_counted)

def extract_elements(form):
    "Build sorted tuple of all elements used in form."
    args = chain(extract_arguments(form), extract_coefficients(form))
    return tuple(f.element() for f in args)

def extract_unique_elements(form):
    "Build sorted tuple of all unique elements used in form."
    return unique_tuple(extract_elements(form))

def extract_sub_elements(elements):
    "Build sorted tuple of all sub elements (including parent element)."
    sub_elements = tuple(chain(*[e.sub_elements() for e in elements if isinstance(e, MixedElement)]))
    if not sub_elements: return tuple(elements)
    return tuple(elements) + extract_sub_elements(sub_elements)

def extract_unique_sub_elements(elements):
    "Build sorted tuple of all unique sub elements (including parent element)."
    return unique_tuple(extract_sub_elements(elements))

def extract_element_map(elements):
    "Build map from elements to element index in ordered tuple."
    element_map = {}
    unique_elements = unique_tuple(elements)
    for element in elements:
        indices = [i for (i, e) in enumerate(unique_elements) if e == element]
        ufl_assert(len(indices) == 1, "Unable to find unique index for element.")
        element_map[element] = i
    return element_map

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

def extract_max_quadrature_element_degree(integral):
    """Extract quadrature integration order from quadrature
    elements in integral. Returns None if not found."""
    quadrature_elements = [e for e in extract_elements(integral) if "Quadrature" in e.family()]
    degrees = [element.degree() for element in quadrature_elements]
    degrees = [q for q in degrees if not q is None]
    if not degrees:
        return None
    max_degree = quadrature_elements[0].degree()
    ufl_assert(all(max_degree == q for q in degrees),
               "Incompatible quadrature elements specified (orders must be equal).")
    return max_degree

def estimate_quadrature_degree(integral):
    "Estimate the necessary quadrature order for integral using the sum of basis function degrees."
    arguments = extract_arguments(integral)
    degrees = [v.element().degree() for v in arguments]
    if len(arguments) == 0:
        return None
    if len(arguments) == 1:
        return 2*degrees[0]
    return sum(degrees)

def unique_tuple(objects):
    "Return sorted tuple of unique objects."
    unique_objects = []
    for object in objects:
        if not object in unique_objects:
            unique_objects.append(object)
    return tuple(unique_objects)

def extract_num_sub_domains(form):
    "Extract the number of sub domains for each domain type."
    num_domains = {}
    for domain_type in _domain_types:
        num_domains[domain_type] = 0
    for integral in form.integrals():
        domain_type = integral.measure().domain_type()
        domain_id = integral.measure().domain_id()
        num_domains[domain_type] = max(num_domains[domain_type], domain_id + 1)
    num_domains = tuple([num_domains[domain_type] for domain_type in _domain_types])
    return num_domains

def extract_integral_data(form):
    """
    Extract integrals from form stored by integral type and sub
    domain, stored as a list of tuples

        (domain_type, domain_id, integrals, metadata)

    where metadata is an empty dictionary that may be used for
    associating metadata with each tuple.
    """

    # Extract integral data
    integral_data = {}
    for integral in form.integrals():
        domain_type = integral.measure().domain_type()
        domain_id = integral.measure().domain_id()
        if (domain_type, domain_id) in integral_data:
            integral_data[(domain_type, domain_id)].append(integral)
        else:
            integral_data[(domain_type, domain_id)] = [integral]

    # Note that this sorting thing here is pretty interesting. The
    # domain types happen to be in alphabetical order (cell, exterior,
    # interior, and macro) so we can just sort... :-)

    # Sort by domain type and number
    integral_data = [(d, n, i, {}) for ((d, n), i) in integral_data.iteritems()]
    integral_data.sort()

    return integral_data

def sort_elements(elements):
    """
    Sort elements so that any sub elements appear before the
    corresponding mixed elements. This is useful when sub elements
    need to be defined before the corresponding mixed elements.

    The ordering is based on sorting a directed acyclic graph.
    """

    # Set nodes
    nodes = elements

    # Set edges
    edges = dict((node, []) for node in nodes)
    for element in elements:
        for sub_element in element.sub_elements():
            edges[element].append(sub_element)

    # Sort graph
    sorted_elements = topological_sorting(nodes, edges)

    # Reverse list of elements
    sorted_elements.reverse()

    return sorted_elements
