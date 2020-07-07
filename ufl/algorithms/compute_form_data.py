# -*- coding: utf-8 -*-
"""This module provides the compute_form_data function which form compilers
will typically call prior to code generation to preprocess/simplify a
raw input form given by a user."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from itertools import chain

from ufl.log import error, info
from ufl.utils.sequences import max_degree
from ufl.measure import custom_integral_types

from ufl.classes import GeometricFacetQuantity, Coefficient, Form, FunctionSpace
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.algorithms.analysis import extract_coefficients, extract_sub_elements, unique_tuple
from ufl.algorithms.formdata import FormData
from ufl.algorithms.formtransformations import compute_form_arities
from ufl.algorithms.check_arities import check_form_arity
from ufl.algorithms.check_restrictions import check_restrictions

# These are the main symbolic processing steps:
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives, apply_coordinate_derivatives
from ufl.algorithms.apply_integral_scaling import apply_integral_scaling
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_restrictions import apply_restrictions, apply_default_restrictions
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms.comparison_checker import do_comparison_check

# See TODOs at the call sites of these below:
from ufl.algorithms.domain_analysis import build_integral_data
from ufl.algorithms.domain_analysis import reconstruct_form_from_integral_data
from ufl.algorithms.domain_analysis import group_form_integrals


def _auto_select_degree(elements):
    """
    Automatically select degree for all elements of the form in cases
    where this has not been specified by the user. This feature is
    used by DOLFIN to allow the specification of Expressions with
    undefined degrees.
    """
    # Use max degree of all elements, at least 1 (to work with
    # Lagrange elements)
    return max_degree({e.degree() for e in elements} - {None} | {1})


def _compute_element_mapping(form):
    "Compute element mapping for element replacement"
    # The element mapping is a slightly messy concept with two use
    # cases:
    # - Expression with missing cell or element TODO: Implement proper
    #   Expression handling in UFL and get rid of this
    # - Constant with missing cell TODO: Fix anything that needs to be
    #   worked around to drop this requirement

    # Extract all elements and include subelements of mixed elements
    elements = [obj.ufl_element() for obj in chain(form.arguments(),
                                                   form.coefficients())]
    elements = extract_sub_elements(elements)

    # Try to find a common degree for elements
    common_degree = _auto_select_degree(elements)

    # Compute element map
    element_mapping = {}
    for element in elements:

        # Flag for whether element needs to be reconstructed
        reconstruct = False

        # Set cell
        cell = element.cell()
        if cell is None:
            domains = form.ufl_domains()
            if not all(domains[0].ufl_cell() == d.ufl_cell()
                       for d in domains):
                error("Cannot replace unknown element cell without unique common cell in form.")
            cell = domains[0].ufl_cell()
            info("Adjusting missing element cell to %s." % (cell,))
            reconstruct = True

        # Set degree
        degree = element.degree()
        if degree is None:
            info("Adjusting missing element degree to %d" % (common_degree,))
            degree = common_degree
            reconstruct = True

        # Reconstruct element and add to map
        if reconstruct:
            element_mapping[element] = element.reconstruct(cell=cell, degree=degree)
        else:
            element_mapping[element] = element

    return element_mapping


def _compute_max_subdomain_ids(integral_data):
    max_subdomain_ids = {}
    for itg_data in integral_data:
        it = itg_data.integral_type
        si = itg_data.subdomain_id
        if isinstance(si, int):
            newmax = si + 1
        else:
            newmax = 0
        prevmax = max_subdomain_ids.get(it, 0)
        max_subdomain_ids[it] = max(prevmax, newmax)
    return max_subdomain_ids


def _compute_form_data_elements(self, arguments, coefficients, domains):
    self.argument_elements = tuple(f.ufl_element() for f in arguments)
    self.coefficient_elements = tuple(f.ufl_element() for f in coefficients)
    self.coordinate_elements = tuple(domain.ufl_coordinate_element() for domain in domains)

    # TODO: Include coordinate elements from argument and coefficient
    # domains as well? Can they differ?

    # Note: Removed self.elements and self.sub_elements to make sure
    #       code that depends on the selection of argument +
    #       coefficient elements blow up, as opposed to silently
    #       almost working, with the introduction of the coordinate
    #       elements here.

    all_elements = self.argument_elements + self.coefficient_elements + self.coordinate_elements
    all_sub_elements = extract_sub_elements(all_elements)

    self.unique_elements = unique_tuple(all_elements)
    self.unique_sub_elements = unique_tuple(all_sub_elements)


def _check_elements(form_data):
    for element in chain(form_data.unique_elements,
                         form_data.unique_sub_elements):
        if element.family() is None:
            error("Found element with undefined family: %s" % repr(element))
        if element.cell() is None:
            error("Found element with undefined cell: %s" % repr(element))


def _check_facet_geometry(integral_data):
    for itg_data in integral_data:
        for itg in itg_data.integrals:
            it = itg_data.integral_type
            # Facet geometry is only valid in facet integrals.
            # Allowing custom integrals to pass as well, although
            # that's not really strict enough.
            if not ("facet" in it or "custom" in it or "interface" in it):
                # Not a facet integral
                for expr in traverse_unique_terminals(itg.integrand()):
                    cls = expr._ufl_class_
                    if issubclass(cls, GeometricFacetQuantity):
                        error("Integral of type %s cannot contain a %s." % (it, cls.__name__))


def _check_form_arity(preprocessed_form):
    # Check that we don't have a mixed linear/bilinear form or
    # anything like that
    # FIXME: This is slooow and should be moved to form compiler
    # and/or replaced with something faster
    if 1 != len(compute_form_arities(preprocessed_form)):
        error("All terms in form must have same rank.")


def _check_restrictions(integral_data):
    # Only allow restrictions on interior facet integrals
    for itg_data in integral_data:
        for itg in itg_data.integrals:
            if itg_data.integral_type not in custom_integral_types:  # Allowing custom integrals to pass
                if itg_data.integral_type.startswith("interior_facet"):
                    check_restrictions(itg.integrand(), True)
                else:
                    check_restrictions(itg.integrand(), False)


def _build_coefficient_replace_map(coefficients, element_mapping=None):
    """Create new Coefficient objects
    with count starting at 0. Return mapping from old
    to new objects, and lists of the new objects."""
    if element_mapping is None:
        element_mapping = {}

    new_coefficients = []
    replace_map = {}
    for i, f in enumerate(coefficients):
        old_e = f.ufl_element()
        new_e = element_mapping.get(old_e, old_e)
        # XXX: This is a hack to ensure that if the original
        # coefficient had a domain, the new one does too.
        # This should be overhauled with requirement that Expressions
        # always have a domain.
        if f.ufl_domain() is not None:
            new_e = FunctionSpace(f.ufl_domain(), new_e)
        new_f = Coefficient(new_e, count=i)
        new_coefficients.append(new_f)
        replace_map[f] = new_f

    return new_coefficients, replace_map


def attach_estimated_degrees(form):
    """Attach estimated polynomial degree to a form's integrals.

    :arg form: The :class:`~.Form` to inspect.
    :returns: A new Form with estimate degrees attached.
    """
    integrals = form.integrals()

    new_integrals = []
    for integral in integrals:
        md = {}
        md.update(integral.metadata())
        degree = estimate_total_polynomial_degree(integral.integrand())
        md["estimated_polynomial_degree"] = degree
        new_integrals.append(integral.reconstruct(metadata=md))
    return Form(new_integrals)


def compute_form_data(form,
                      # Default arguments configured to behave the way old FFC expects it:
                      do_apply_function_pullbacks=False,
                      do_apply_integral_scaling=False,
                      do_apply_geometry_lowering=False,
                      preserve_geometry_types=(),
                      do_apply_default_restrictions=True,
                      do_apply_restrictions=True,
                      do_estimate_degrees=True,
                      do_append_everywhere_integrals=True,
                      complex_mode=False,
                      ):

    # TODO: Move this to the constructor instead
    self = FormData()

    # --- Store untouched form for reference.
    # The user of FormData may get original arguments,
    # original coefficients, and form signature from this object.
    # But be aware that the set of original coefficients are not
    # the same as the ones used in the final UFC form.
    # See 'reduced_coefficients' below.
    self.original_form = form

    # --- Pass form integrands through some symbolic manipulation

    # Note: Default behaviour here will process form the way that is
    # currently expected by vanilla FFC

    # Check that the form does not try to compare complex quantities:
    # if the quantites being compared are 'provably' real, wrap them
    # with Real, otherwise throw an error.
    if complex_mode:
        form = do_comparison_check(form)

    # Lower abstractions for tensor-algebra types into index notation,
    # reducing the number of operators later algorithms and form
    # compilers need to handle
    form = apply_algebra_lowering(form)

    # After lowering to index notation, remove any complex nodes that
    # have been introduced but are not wanted when working in real mode,
    # allowing for purely real forms to be written
    if not complex_mode:
        form = remove_complex_nodes(form)

    # Apply differentiation before function pullbacks, because for
    # example coefficient derivatives are more complicated to derive
    # after coefficients are rewritten, and in particular for
    # user-defined coefficient relations it just gets too messy
    form = apply_derivatives(form)

    # --- Group form integrals
    # TODO: Refactor this, it's rather opaque what this does
    # TODO: Is self.original_form.ufl_domains() right here?
    #       It will matter when we start including 'num_domains' in ufc form.
    form = group_form_integrals(form, self.original_form.ufl_domains(),
                                do_append_everywhere_integrals=do_append_everywhere_integrals)

    # Estimate polynomial degree of integrands now, before applying
    # any pullbacks and geometric lowering.  Otherwise quad degrees
    # blow up horrifically.
    if do_estimate_degrees:
        form = attach_estimated_degrees(form)

    if do_apply_function_pullbacks:
        # Rewrite coefficients and arguments in terms of their
        # reference cell values with Piola transforms and symmetry
        # transforms injected where needed.
        # Decision: Not supporting grad(dolfin.Expression) without a
        #           Domain.  Current dolfin works if Expression has a
        #           cell but this should be changed to a mesh.
        form = apply_function_pullbacks(form)

    # Scale integrals to reference cell frames
    if do_apply_integral_scaling:
        form = apply_integral_scaling(form)

    # Apply default restriction to fully continuous terminals
    if do_apply_default_restrictions:
        form = apply_default_restrictions(form)

    # Lower abstractions for geometric quantities into a smaller set
    # of quantities, allowing the form compiler to deal with a smaller
    # set of types and treating geometric quantities like any other
    # expressions w.r.t. loop-invariant code motion etc.
    if do_apply_geometry_lowering:
        form = apply_geometry_lowering(form, preserve_geometry_types)

    # Apply differentiation again, because the algorithms above can
    # generate new derivatives or rewrite expressions inside
    # derivatives
    if do_apply_function_pullbacks or do_apply_geometry_lowering:
        form = apply_derivatives(form)

        # Neverending story: apply_derivatives introduces new Jinvs,
        # which needs more geometry lowering
        if do_apply_geometry_lowering:
            form = apply_geometry_lowering(form, preserve_geometry_types)
            # Lower derivatives that may have appeared
            form = apply_derivatives(form)

    form = apply_coordinate_derivatives(form)

    # Propagate restrictions to terminals
    if do_apply_restrictions:
        form = apply_restrictions(form)

    # If in real mode, remove any complex nodes introduced during form processing.
    if not complex_mode:
        form = remove_complex_nodes(form)

    # --- Group integrals into IntegralData objects
    # Most of the heavy lifting is done above in group_form_integrals.
    self.integral_data = build_integral_data(form.integrals())

    # --- Create replacements for arguments and coefficients

    # Figure out which form coefficients each integral should enable
    for itg_data in self.integral_data:
        itg_coeffs = set()
        # Get all coefficients in integrand
        for itg in itg_data.integrals:
            itg_coeffs.update(extract_coefficients(itg.integrand()))
        # Store with IntegralData object
        itg_data.integral_coefficients = itg_coeffs

    # Figure out which coefficients from the original form are
    # actually used in any integral (Differentiation may reduce the
    # set of coefficients w.r.t. the original form)
    reduced_coefficients_set = set()
    for itg_data in self.integral_data:
        reduced_coefficients_set.update(itg_data.integral_coefficients)
    self.reduced_coefficients = sorted(reduced_coefficients_set,
                                       key=lambda c: c.count())
    self.num_coefficients = len(self.reduced_coefficients)
    self.original_coefficient_positions = [i for i, c in enumerate(self.original_form.coefficients())
                                           if c in self.reduced_coefficients]

    # Store back into integral data which form coefficients are used
    # by each integral
    for itg_data in self.integral_data:
        itg_data.enabled_coefficients = [bool(coeff in itg_data.integral_coefficients)
                                         for coeff in self.reduced_coefficients]

    # --- Collect some trivial data

    # Get rank of form from argument list (assuming not a mixed arity form)
    self.rank = len(self.original_form.arguments())

    # Extract common geometric dimension (topological is not common!)
    self.geometric_dimension = self.original_form.integrals()[0].ufl_domain().geometric_dimension()

    # --- Build mapping from old incomplete element objects to new
    # well defined elements.  This is to support the Expression
    # construct in dolfin which subclasses Coefficient but doesn't
    # provide an element, and the Constant construct that doesn't
    # provide the domain that a Coefficient is supposed to have. A
    # future design iteration in UFL/UFC/FFC/DOLFIN may allow removal
    # of this mapping with the introduction of UFL types for
    # Expression-like functions that can be evaluated in quadrature
    # points.
    self.element_replace_map = _compute_element_mapping(self.original_form)

    # Mappings from elements and coefficients that reside in form to
    # objects with canonical numbering as well as completed cells and
    # elements
    renumbered_coefficients, function_replace_map = \
        _build_coefficient_replace_map(self.reduced_coefficients,
                                       self.element_replace_map)
    self.function_replace_map = function_replace_map

    # --- Store various lists of elements and sub elements (adds
    #     members to self)
    _compute_form_data_elements(self,
                                self.original_form.arguments(),
                                renumbered_coefficients,
                                self.original_form.ufl_domains())

    # --- Store number of domains for integral types
    # TODO: Group this by domain first. For now keep a backwards
    # compatible data structure.
    self.max_subdomain_ids = _compute_max_subdomain_ids(self.integral_data)

    # --- Checks
    _check_elements(self)
    _check_facet_geometry(self.integral_data)
    _check_restrictions(self.integral_data)

    # TODO: This is a very expensive check... Replace with something
    # faster!
    preprocessed_form = reconstruct_form_from_integral_data(self.integral_data)

    check_form_arity(preprocessed_form, self.original_form.arguments(), complex_mode)  # Currently testing how fast this is

    # TODO: This member is used by unit tests, change the tests to
    # remove this!
    self.preprocessed_form = preprocessed_form

    return self
