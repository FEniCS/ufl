# -*- coding: utf-8 -*-
"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from itertools import chain
from six.moves import zip
from collections import namedtuple

from ufl.log import error, warning, info
from ufl.assertions import ufl_assert
from ufl.utils.sorting import sorted_by_count, topological_sorting

from ufl.core.expr import Expr
from ufl.core.terminal import Terminal, FormArgument
from ufl.finiteelement import MixedElement, RestrictedElement
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.variable import Variable
from ufl.core.multiindex import Index, MultiIndex
from ufl.geometry import Domain
from ufl.integral import Measure, Integral
from ufl.form import Form
from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import pre_traversal, traverse_terminals


# TODO: Some of these can possibly be optimised by implementing inlined stack based traversal algorithms


def _sorted_by_number_and_part(seq):
    return sorted(seq, key=lambda x: (x.number(), x.part()))


def unique_tuple(objects):
    "Return tuple of unique objects, preserving initial ordering."
    unique_objects = []
    handled = set()
    for obj in objects:
        if obj not in handled:
            handled.add(obj)
            unique_objects.append(obj)
    return tuple(unique_objects)


#--- Utilities to extract information from an expression ---

def __unused__extract_classes(a):
    """Build a set of all unique Expr subclasses used in a.
    The argument a can be a Form, Integral or Expr."""
    return set(o._ufl_class_
               for e in iter_expressions(a)
               for o in pre_traversal(e))

def extract_type(a, ufl_type):
    """Build a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or Expr."""
    if issubclass(ufl_type, Terminal):
        # Optimization
        return set(o for e in iter_expressions(a)
                   for o in traverse_terminals(e)
                   if isinstance(o, ufl_type))
    else:
        return set(o for e in iter_expressions(a)
                   for o in pre_traversal(e)
                   if isinstance(o, ufl_type))

def has_type(a, ufl_type):
    """Return if an object of class ufl_type can be found in a.
    The argument a can be a Form, Integral or Expr."""
    if issubclass(ufl_type, Terminal):
        # Optimization
        traversal = traverse_terminals
    else:
        traversal = pre_traversal
    return any(isinstance(o, ufl_type) for e in iter_expressions(a) for o in traversal(e))

def has_exact_type(a, ufl_type):
    """Return if an object of class ufl_type can be found in a.
    The argument a can be a Form, Integral or Expr."""
    tc = ufl_type._ufl_typecode_
    if issubclass(ufl_type, Terminal):
        # Optimization
        traversal = traverse_terminals
    else:
        traversal = pre_traversal
    return any(o._ufl_typecode_ == tc for e in iter_expressions(a) for o in traversal(e))

def extract_arguments(a):
    """Build a sorted list of all arguments in a,
    which can be a Form, Integral or Expr."""
    return _sorted_by_number_and_part(extract_type(a, Argument))

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

    # Build number,part: instance mappings, should be one to one
    bfnp = dict((f, (f.number(), f.part())) for f in arguments)
    if len(bfnp) != len(set(bfnp.values())):
        msg = """\
Found different Arguments with same number and part.
Did you combine test or trial functions from different spaces?
The Arguments found are:\n%s""" % "\n".join("  %s" % f for f in arguments)
        error(msg)

    # Build count: instance mappings, should be one to one
    fcounts = dict((f, f.count()) for f in coefficients)
    if len(fcounts) != len(set(fcounts.values())):
        msg = """\
Found different coefficients with same counts.
The arguments found are:\n%s""" % "\n".join("  %s" % f for f in coefficients)
        error(msg)

    # Passed checks, so we can safely sort the instances by count
    arguments = _sorted_by_number_and_part(arguments)
    coefficients = sorted_by_count(coefficients)

    return arguments, coefficients


def extract_elements(form):
    "Build sorted tuple of all elements used in form."
    args = chain(*extract_arguments_and_coefficients(form))
    return tuple(f.element() for f in args)


def extract_unique_elements(form):
    "Build sorted tuple of all unique elements used in form."
    return unique_tuple(extract_elements(form))


def extract_sub_elements(elements):
    "Build sorted tuple of all sub elements (including parent element)."
    sub_elements = tuple(chain(*[e.sub_elements() for e in elements]))
    if not sub_elements: return tuple(elements)
    return tuple(elements) + extract_sub_elements(sub_elements)


def __unused__extract_unique_sub_elements(elements):
    "Build sorted tuple of all unique sub elements (including parent element)."
    return unique_tuple(extract_sub_elements(elements))


def __unused__extract_element_map(elements):
    "Build map from elements to element index in ordered tuple."
    element_map = {}
    unique_elements = unique_tuple(elements)
    for element in elements:
        indices = [i for (i, e) in enumerate(unique_elements) if e == element]
        ufl_assert(len(indices) == 1, "Unable to find unique index for element.")
        element_map[element] = i
    return element_map


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
