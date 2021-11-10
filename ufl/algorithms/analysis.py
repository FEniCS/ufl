# -*- coding: utf-8 -*-
"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010.
# Modified by Johan Hake, 2010.

from itertools import chain

from ufl.log import error
from ufl.utils.sorting import sorted_by_count, topological_sorting

from ufl.core.terminal import Terminal
from ufl.argument import BaseArgument
from ufl.coefficient import BaseCoefficient
from ufl.constant import Constant
from ufl.form import BaseForm, Form
from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import unique_pre_traversal, traverse_unique_terminals
from ufl.core.external_operator import ExternalOperator


# TODO: Some of these can possibly be optimised by implementing
# inlined stack based traversal algorithms

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


# --- Utilities to extract information from an expression ---

def __unused__extract_classes(a):
    """Build a set of all unique Expr subclasses used in a.
    The argument a can be a BaseForm, Integral or Expr."""
    return set(o._ufl_class_
               for e in iter_expressions(a)
               for o in unique_pre_traversal(e))


def extract_type(a, ufl_types):
    """Build a set of all objects found in a whose class is in ufl_types.
    The argument a can be a BaseForm, Integral or Expr."""

    if not isinstance(ufl_types, (list, tuple)):
        ufl_types = (ufl_types,)

    # BaseForms that aren't forms or external operators only have arguments
    if isinstance(a, BaseForm) and not isinstance(a, (Form, ExternalOperator)):
        if any(issubclass(t, BaseArgument) for t in ufl_types):
            return set(a.arguments())
        else:
            return set()

    if all(issubclass(t, Terminal) for t in ufl_types):
        # Optimization
        objects = set(o for e in iter_expressions(a)
                      for o in traverse_unique_terminals(e)
                      if any(isinstance(o, t) for t in ufl_types))
    else:
        objects = set(o for e in iter_expressions(a)
                      for o in unique_pre_traversal(e)
                      if any(isinstance(o, t) for t in ufl_types))

    # Need to extract objects contained in external operators whose type is in ufl_types
    if all(t is not ExternalOperator for t in ufl_types):
        # For the case where there are no external operators in `a`, this effectively doubles the time
        # since we traverse the DAG twice.
        # -> A solution would be to have a mechanism to collect external operators as we traverse
        #    the DAG for the first time and use that information.
        extops = extract_type(a, ExternalOperator)
    else:
        extops = set(e for e in objects if isinstance(e, ExternalOperator))

    ufl_types_no_args = tuple(t for t in ufl_types if not issubclass(t, BaseArgument))
    extop_objects = ()
    for o in extops:
        # Look for BaseArguments in ExternalOperator's argument slots only since that's where they are by definition.
        # Don't look into operands, which is convenient for external operator composition, e.g. N1(N2; v*)
        # where N2 is seen as an operator and not a form.
        for ai in tuple(arg for arg in o.argument_slots(isinstance(a, Form))):
            extop_objects += tuple(extract_type(ai, ufl_types))
        slots = o.ufl_operands + (o.result_coefficient(),)
        for ai in slots:
            extop_objects += tuple(extract_type(ai, ufl_types_no_args))
    objects.update(extop_objects)
    return objects


def has_type(a, ufl_type):
    """Return if an object of class ufl_type can be found in a.
    The argument a can be a BaseForm, Integral or Expr."""
    if issubclass(ufl_type, Terminal):
        # Optimization
        traversal = traverse_unique_terminals
    else:
        traversal = unique_pre_traversal
    return any(isinstance(o, ufl_type) for e in iter_expressions(a) for o in traversal(e))


def has_exact_type(a, ufl_type):
    """Return if an object of class ufl_type can be found in a.
    The argument a can be a BaseForm, Integral or Expr."""
    tc = ufl_type._ufl_typecode_
    if issubclass(ufl_type, Terminal):
        # Optimization
        traversal = traverse_unique_terminals
    else:
        traversal = unique_pre_traversal
    return any(o._ufl_typecode_ == tc for e in iter_expressions(a) for o in traversal(e))


def extract_arguments(a):
    """Build a sorted list of all arguments in a,
    which can be a BaseForm, Integral or Expr."""
    return _sorted_by_number_and_part(extract_type(a, BaseArgument))


def extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a BaseForm, Integral or Expr."""
    return sorted_by_count(extract_type(a, BaseCoefficient))


def extract_external_operators(a):
    """Build a sorted list of all external operators in a,
    which can be a Form, Integral or Expr."""
    return sorted_by_count(extract_type(a, ExternalOperator))


def extract_constants(a):
    """Build a sorted list of all constants in a"""
    return sorted_by_count(extract_type(a, Constant))


def extract_arguments_and_coefficients(a):
    """Build two sorted lists of all arguments and coefficients
    in a, which can be BaseForm, Integral or Expr."""

    # This function is faster than extract_arguments + extract_coefficients
    # for large forms, and has more validation built in.

    # Extract lists of all BaseArgument and BaseCoefficient instances
    base_coeff_and_args = extract_type(a, (BaseArgument, BaseCoefficient))
    arguments = [f for f in base_coeff_and_args if isinstance(f, BaseArgument)]
    coefficients = [f for f in base_coeff_and_args if isinstance(f, BaseCoefficient)]

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
    return tuple(f.ufl_element() for f in args)


def extract_unique_elements(form):
    "Build sorted tuple of all unique elements used in form."
    return unique_tuple(extract_elements(form))


def extract_sub_elements(elements):
    "Build sorted tuple of all sub elements (including parent element)."
    sub_elements = tuple(chain(*[e.sub_elements() for e in elements]))
    if not sub_elements:
        return tuple(elements)
    return tuple(elements) + extract_sub_elements(sub_elements)


def sort_elements(elements):
    """
    Sort elements so that any sub elements appear before the
    corresponding mixed elements. This is useful when sub elements
    need to be defined before the corresponding mixed elements.

    The ordering is based on sorting a directed acyclic graph.
    """

    # Set nodes
    nodes = sorted(elements)

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
