"""Utility algorithms for inspection of and information extraction from UFL objects."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010.
# Modified by Johan Hake, 2010.

from itertools import chain

from ufl.algorithms.traversal import iter_expressions
from ufl.argument import BaseArgument, Coargument
from ufl.coefficient import BaseCoefficient
from ufl.constant import Constant
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.terminal import Terminal
from ufl.corealg.traversal import traverse_unique_terminals, unique_pre_traversal
from ufl.domain import Mesh
from ufl.form import BaseForm, Form
from ufl.geometry import GeometricQuantity
from ufl.utils.sorting import sorted_by_count, topological_sorting

# TODO: Some of these can possibly be optimised by implementing
# inlined stack based traversal algorithms


def _sorted_by_number_and_part(seq):
    """Sort items by number and part."""
    return sorted(seq, key=lambda x: (x.number(), x.part()))


def unique_tuple(objects):
    """Return tuple of unique objects, preserving initial ordering."""
    unique_objects = []
    handled = set()
    for obj in objects:
        if obj not in handled:
            handled.add(obj)
            unique_objects.append(obj)
    return tuple(unique_objects)


# --- Utilities to extract information from an expression ---


def extract_type(a, ufl_types):
    """Build a set of all objects found in a whose class is in ufl_types.

    Args:
        a: A BaseForm, Integral or Expr
        ufl_types: A list of UFL types

    Returns:
        All objects found in a whose class is in ufl_type
    """
    if not isinstance(ufl_types, (list, tuple)):
        ufl_types = (ufl_types,)

    if all(t is not BaseFormOperator for t in ufl_types):
        remove_base_form_ops = True
        ufl_types += (BaseFormOperator,)
    else:
        remove_base_form_ops = False

    # BaseForms that aren't forms or base form operators
    # only contain arguments & coefficients
    if isinstance(a, BaseForm) and not isinstance(a, (Form, BaseFormOperator)):
        objects = set()
        arg_types = tuple(t for t in ufl_types if issubclass(t, BaseArgument))
        if arg_types:
            objects.update(e for e in a.arguments() if isinstance(e, arg_types))
        coeff_types = tuple(t for t in ufl_types if issubclass(t, BaseCoefficient))
        if coeff_types:
            objects.update(e for e in a.coefficients() if isinstance(e, coeff_types))
        return objects

    if all(issubclass(t, Terminal) for t in ufl_types):
        # Optimization
        traversal = traverse_unique_terminals
    else:
        traversal = unique_pre_traversal

    objects = set(o for e in iter_expressions(a) for o in traversal(e) if isinstance(o, ufl_types))

    # Need to extract objects contained in base form operators whose
    # type is in ufl_types
    base_form_ops = set(e for e in objects if isinstance(e, BaseFormOperator))
    ufl_types_no_args = tuple(t for t in ufl_types if not issubclass(t, BaseArgument))
    base_form_objects = []
    for o in base_form_ops:
        # This accounts for having BaseFormOperator in Forms: if N is a BaseFormOperator
        # `N(u; v*) * v * dx` <=> `action(v1 * v * dx, N(...; v*))`
        # where `v`, `v1` are `Argument`s and `v*` a `Coargument`.
        for ai in tuple(arg for arg in o.argument_slots(isinstance(a, Form))):
            # Extracting BaseArguments of an object of which a
            # Coargument is an argument, then we just return the dual
            # argument of the Coargument and not its primal argument.
            if isinstance(ai, Coargument):
                new_types = tuple(Coargument if t is BaseArgument else t for t in ufl_types)
                base_form_objects.extend(extract_type(ai, new_types))
            else:
                base_form_objects.extend(extract_type(ai, ufl_types))
        # Look for BaseArguments in BaseFormOperator's argument slots
        # only since that's where they are by definition. Don't look
        # into operands, which is convenient for external operator
        # composition, e.g. N1(N2; v*) where N2 is seen as an operator
        # and not a form.
        slots = o.ufl_operands
        for ai in slots:
            base_form_objects.extend(extract_type(ai, ufl_types_no_args))
    objects.update(base_form_objects)

    # `Remove BaseFormOperator` objects if there were initially not in `ufl_types`
    if remove_base_form_ops:
        objects -= base_form_ops
    return objects


def has_type(a, ufl_type):
    """Return if an object of class ufl_type or a subclass can be found in a.

    Args:
        a: A BaseForm, Integral or Expr
        ufl_type: A UFL type

    Returns:
        Whether an object of class ufl_type can be found in a
    """
    if issubclass(ufl_type, Terminal):
        # Optimization
        traversal = traverse_unique_terminals
    else:
        traversal = unique_pre_traversal
    return any(isinstance(o, ufl_type) for e in iter_expressions(a) for o in traversal(e))


def has_exact_type(a, ufl_type):
    """Return if an object of class ufl_type can be found in a.

    Args:
        a: A BaseForm, Integral or Expr
        ufl_type: A UFL type

    Returns:
        Whether an object of class ufl_type can be found in a
    """
    tc = ufl_type._ufl_typecode_
    if issubclass(ufl_type, Terminal):
        # Optimization
        traversal = traverse_unique_terminals
    else:
        traversal = unique_pre_traversal
    return any(o._ufl_typecode_ == tc for e in iter_expressions(a) for o in traversal(e))


def extract_arguments(a):
    """Build a sorted list of all arguments in a.

    Args:
        a: A BaseForm, Integral or Expr
    """
    return _sorted_by_number_and_part(extract_type(a, BaseArgument))


def extract_coefficients(a):
    """Build a sorted list of all coefficients in a.

    Args:
        a: A BaseForm, Integral or Expr
    """
    return sorted_by_count(extract_type(a, BaseCoefficient))


def extract_constants(a):
    """Build a sorted list of all constants in a.

    Args:
        a: A BaseForm, Integral or Expr
    """
    return sorted_by_count(extract_type(a, Constant))


def extract_base_form_operators(a):
    """Build a sorted list of all base form operators in a.

    Args:
        a: A BaseForm, Integral or Expr
    """
    return sorted_by_count(extract_type(a, BaseFormOperator))


def extract_terminals_with_domain(a):
    """Build three sorted lists of all arguments, coefficients, and geometric quantities in `a`.

    This function is faster than extracting each type of terminal
    separately for large forms, and has more validation built in.

    Args:
        a: A BaseForm, Integral or Expr

    Returns:
        Tuples of extracted `Argument`s, `Coefficient`s, and `GeometricQuantity`s.

    """
    # Extract lists of all BaseArgument, BaseCoefficient, and GeometricQuantity instances
    terminals = extract_type(a, (BaseArgument, BaseCoefficient, GeometricQuantity))
    arguments = [f for f in terminals if isinstance(f, BaseArgument)]
    coefficients = [f for f in terminals if isinstance(f, BaseCoefficient)]
    geometric_quantities = [f for f in terminals if isinstance(f, GeometricQuantity)]

    # Build number,part: instance mappings, should be one to one
    bfnp = {f: (f.number(), f.part()) for f in arguments}
    if len(bfnp) != len(set(bfnp.values())):
        raise ValueError(
            "Found different Arguments with same number and part.\n"
            "Did you combine test or trial functions from different spaces?\n"
            "The Arguments found are:\n" + "\n".join(f"  {a}" for a in arguments)
        )

    # Build count: instance mappings, should be one to one
    fcounts = {f: f.count() for f in coefficients}
    if len(fcounts) != len(set(fcounts.values())):
        raise ValueError(
            "Found different coefficients with same counts.\n"
            "The Coefficients found are:\n" + "\n".join(f"  {c}" for c in coefficients)
        )

    # Build count: instance mappings, should be one to one
    gqcounts = {}
    for gq in geometric_quantities:
        if not isinstance(gq._domain, Mesh):
            raise TypeError(f"{gq}._domain must be a Mesh: got {gq._domain}")
        gqcounts[gq] = (type(gq).name, gq._domain._ufl_id)
    if len(gqcounts) != len(set(gqcounts.values())):
        raise ValueError(
            "Found different geometric quantities with same (geometric_quantity_type, domain).\n"
            "The GeometricQuantities found are:\n"
            "\n".join(f"  {gq}" for gq in geometric_quantities)
        )

    # Passed checks, so we can safely sort the instances by count
    arguments = _sorted_by_number_and_part(arguments)
    coefficients = sorted_by_count(coefficients)
    geometric_quantities = list(
        sorted(geometric_quantities, key=lambda gq: (type(gq).name, gq._domain._ufl_id))
    )

    return arguments, coefficients, geometric_quantities


def extract_elements(form):
    """Build sorted tuple of all elements used in form."""
    arguments, coefficients, _ = extract_terminals_with_domain(form)
    return tuple(f.ufl_element() for f in arguments + coefficients)


def extract_unique_elements(form):
    """Build sorted tuple of all unique elements used in form."""
    return unique_tuple(extract_elements(form))


def extract_sub_elements(elements):
    """Build sorted tuple of all sub elements (including parent element)."""
    sub_elements = tuple(chain(*(e.sub_elements for e in elements)))
    if not sub_elements:
        return tuple(elements)
    return (*elements, *extract_sub_elements(sub_elements))


def sort_elements(elements):
    """Sort elements.

    A sort is performed so that any sub elements appear before the
    corresponding mixed elements. This is useful when sub elements
    need to be defined before the corresponding mixed elements.

    The ordering is based on sorting a directed acyclic graph.
    """
    # Set nodes
    nodes = list(elements)

    # Set edges
    edges = {node: [] for node in nodes}
    for element in elements:
        for sub_element in element.sub_elements:
            edges[element].append(sub_element)

    # Sort graph
    sorted_elements = topological_sorting(nodes, edges)

    # Reverse list of elements
    sorted_elements.reverse()

    return sorted_elements
