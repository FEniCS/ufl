"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

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
# First added:  2008-03-14
# Last changed: 2013-01-02

from itertools import izip, chain
from collections import namedtuple

from ufl.log import error, warning, info
from ufl.assertions import ufl_assert
from ufl.sorting import topological_sorting
from ufl.common import sorted_by_count

from ufl.expr import Expr
from ufl.terminal import Terminal, FormArgument
from ufl.finiteelement import MixedElement, RestrictedElement
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.variable import Variable
from ufl.indexing import Index, MultiIndex
from ufl.domains import Region, Domain
from ufl.integral import Measure, Integral
from ufl.form import Form
from ufl.algorithms.traversal import iter_expressions, post_traversal, post_walk, traverse_terminals

# Domain types (should probably be listed somewhere else)
_domain_types = Measure._domain_types_tuple

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

def extract_arguments(a):
    """Build a sorted list of all arguments in a,
    which can be a Form, Integral or Expr."""
    return sorted_by_count(extract_type(a, Argument))

def extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    return sorted_by_count(extract_type(a, Coefficient))

def extract_arguments_and_coefficients(a):
    """Build two sorted lists of all arguments and coefficients
    in a, which can be a Form, Integral or Expr."""

    # This function is faster than extract_arguments + extract_coefficients
    # for large forms, and has more validation built in.

    # Extract lists of all form argument instances
    terminals = extract_type(a, FormArgument)
    arguments = [f for f in terminals if isinstance(f, Argument)]
    coefficients = [f for f in terminals if isinstance(f, Coefficient)]

    # Build count: instance mappings, should be one to one
    bfcounts = dict((f, f.count()) for f in arguments)
    fcounts = dict((f, f.count()) for f in coefficients)

    if len(bfcounts) != len(set(bfcounts.values())):
        msg = """\
Found different Arguments with same counts.
Did you combine test or trial functions from different spaces?
The Arguments found are:\n%s""" % "\n".join("  %s" % f for f in arguments)
        error(msg)

    if len(fcounts) != len(set(fcounts.values())):
        msg = """\
Found different coefficients with same counts.
The arguments found are:\n%s""" % "\n".join("  %s" % f for f in coefficients)
        error(msg)

    # Passed checks, so we can safely sort the instances by count
    arguments = sorted_by_count(arguments)
    coefficients = sorted_by_count(coefficients)

    return arguments, coefficients

def build_argument_replace_map(arguments, coefficients, element_mapping=None):
    """Create new Argument and Coefficient objects
    with count starting at 0. Return mapping from old
    to new objects, and lists of the new objects."""
    if element_mapping is None:
        element_mapping = {}
    def remap(args):
        for (i, f) in enumerate(args):
            old_e = f.element()
            new_e = element_mapping.get(old_e, old_e)
            yield f.reconstruct(element=new_e, count=i)
    new_arguments = list(remap(arguments))
    new_coefficients = list(remap(coefficients))
    replace_map = dict(izip(chain(arguments, coefficients),
                            chain(new_arguments, new_coefficients)))
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
    return sorted_by_count(s)

def extract_elements(form):
    "Build sorted tuple of all elements used in form."
    args = chain(extract_arguments(form), extract_coefficients(form))
    return tuple(f.element() for f in args)

def extract_unique_elements(form):
    "Build sorted tuple of all unique elements used in form."
    return unique_tuple(extract_elements(form))

def extract_sub_elements(elements):
    "Build sorted tuple of all sub elements (including parent element)."
    sub_elements = tuple(chain(*[e.sub_elements() for e in elements]))
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
    "Estimate the necessary quadrature order for integral using the sum of argument degrees."
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

def extract_domain_data(form):
    "Extract the domain_data attached to integrals of each domain type in form."
    domain_data = {}
    for integral in form.integrals():
        domain_type = integral.domain_type()
        data = integral.domain_data()
        # Check that there is only one domain_data object for each integral type
        existing_data = domain_data.get(domain_type)
        if existing_data is None:
            # Got no data before, store this one. May be None, that's fine.
            domain_data[domain_type] = data
        elif data is None:
            # Already got data, getting no data is ok but don't overwrite.
            pass
        elif existing_data is not data:
            # NB! Using 'is' because we're not assuming anything about domain_data type, not even an equals operator!
            error("Found two domain data objects for same domain type '%s', only one is allowed." % str(domain_type))
    return domain_data

def extract_num_sub_domains(form):
    "Extract the upper limit of sub domain ids for each domain type."
    num_domains = {}
    for integral in form.integrals():
        domain_type = integral.domain_type()
        domain_id = integral.domain_id()

        # TODO: This may need some redesign
        max_domain_id = None
        if isinstance(domain_id, int):
            max_domain_id = domain_id
        elif isinstance(domain_id, Region):
            max_domain_id = max(domain_id.subdomain_ids())
        elif isinstance(domain_id, Domain):
            max_domain_id = None

        if max_domain_id is not None:
            num_domains[domain_type] = max(num_domains.get(domain_type, 0), max_domain_id + 1)
    return num_domains

class IntegralData(object):
    """Utility class with the members
        (domain_type, domain_id, integrals, metadata)

    where metadata is an empty dictionary that may be used for
    associating metadata with each object.
    """
    __slots__ = ('domain_type', 'domain_id', 'integrals', 'metadata')
    def __init__(self, domain_type, domain_id, integrals, metadata):
        self.domain_type = domain_type
        self.domain_id = domain_id
        self.integrals = integrals
        self.metadata = metadata

    def __lt__(self, other):
        # To preserve behaviour of extract_integral_data:
        return ((self.domain_type, self.domain_id, self.integrals, self.metadata)
                < (other.domain_type, other.domain_id, other.integrals, other.metadata))

    def __eq__(self, other):
        # Currently only used for tests:
        return (self.domain_type == other.domain_type and
                self.domain_id == other.domain_id and
                self.integrals == other.integrals and
                self.metadata == other.metadata)

    def __str__(self):
        return "IntegralData object over domain (%s, %s), with integrals:\n%s\nand metadata:\n%s" % (
            self.domain_type, self.domain_id,
            '\n\n'.join(map(str,self.integrals)), self.metadata)

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
