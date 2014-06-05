"""This module provides the compute_form_data function which form compilers
will typically call prior to code generation to preprocess/simplify a
raw input form given by a user."""

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from collections import defaultdict
from itertools import chain
from time import time
import ufl
from ufl.common import lstr, tstr, estr, istr, slice_dict
from ufl.common import Timer
from ufl.assertions import ufl_assert
from ufl.log import error, warning, info
from ufl.expr import Expr
from ufl.form import Form
from ufl.protocols import id_or_none
from ufl.geometry import as_domain
from ufl.classes import GeometricFacetQuantity
from ufl.algorithms.replace import replace
from ufl.algorithms.analysis import (extract_arguments_and_coefficients,
                                     extract_coefficients,
                                     extract_classes,
                                     build_coefficient_replace_map,
                                     extract_elements, extract_sub_elements,
                                     unique_tuple)
from ufl.algorithms.domain_analysis import build_integral_data, reconstruct_form_from_integral_data
from ufl.algorithms.formdata import FormData, ExprData
from ufl.algorithms.expand_indices import expand_indices
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.propagate_restrictions import propagate_restrictions
from ufl.algorithms.formtransformations import compute_form_arities
from ufl.algorithms.signature import compute_expression_signature, compute_form_signature


def _auto_select_degree(elements):
    """
    Automatically select degree for all elements of the form in cases
    where this has not been specified by the user. This feature is
    used by DOLFIN to allow the specification of Expressions with
    undefined degrees.
    """

    # Use max degree of all elements
    common_degree = max([e.degree() for e in elements] or [None])

    # Default to linear element if no elements with degrees are provided
    if common_degree is None:
        common_degree = 1

    # Degree must be at least 1 (to work with Lagrange elements)
    common_degree = max(1, common_degree)

    return common_degree

def _compute_element_mapping(form):
    "Compute element mapping for element replacement"

    # Extract all elements and include subelements of mixed elements
    elements = [obj.element() for obj in chain(form.arguments(), form.coefficients())]
    elements = extract_sub_elements(elements)

    # Try to find a common degree for elements
    common_degree = _auto_select_degree(elements)

    # Compute element map
    element_mapping = {}
    for element in elements:

        # Flag for whether element needs to be reconstructed
        reconstruct = False

        # Set domain/cell
        domain = element.domain()
        if domain is None:
            domains = form.domains()
            ufl_assert(len(domains) == 1,
                       "Cannot replace unknown element domain without unique common domain in form.")
            domain, = domains
            info("Adjusting missing element domain to %s." % (domain,))
            reconstruct = True

        # Set degree
        degree = element.degree()
        if degree is None:
            info("Adjusting missing element degree to %d" % (common_degree,))
            degree = common_degree
            reconstruct = True

        # Reconstruct element and add to map
        if reconstruct:
            element_mapping[element] = element.reconstruct(domain=domain, degree=degree)
        else:
            element_mapping[element] = element

    return element_mapping

def _compute_num_sub_domains(integral_data):
    num_sub_domains = {}
    for itg_data in integral_data:
        it = itg_data.integral_type
        si = itg_data.subdomain_id
        if isinstance(si, str):
            new = 0
        else:
            new = si + 1
        prev = num_sub_domains.get(it)
        num_sub_domains[it] = max(prev, new)
    return num_sub_domains

def _compute_form_data_elements(self, arguments, coefficients):
    self.argument_elements    = tuple(f.element() for f in arguments)
    self.coefficient_elements = tuple(f.element() for f in coefficients)
    self.elements             = self.argument_elements + self.coefficient_elements
    self.unique_elements      = unique_tuple(self.elements)
    self.sub_elements         = extract_sub_elements(self.elements)
    self.unique_sub_elements  = unique_tuple(self.sub_elements)

def _check_elements(form_data):
    for element in chain(form_data.unique_elements, form_data.unique_sub_elements):
        ufl_assert(element.domain() is not None,
                   "Found element with undefined domain: %s" % repr(element))
        ufl_assert(element.family() is not None,
                   "Found element with undefined familty: %s" % repr(element))

def _check_facet_geometry(integral_data):
    for itg_data in integral_data:
        for itg in itg_data.integrals:
            classes = extract_classes(itg.integrand())
            it = itg_data.integral_type
            # Facet geometry is only valid in facet integrals
            if "facet" not in it and "custom" not in it:
                for c in classes:
                    ufl_assert(not issubclass(c, GeometricFacetQuantity),
                               "Integral of type %s cannot contain a %s." % (it, c.__name__))

def _check_form_arity(preprocessed_form):
    # Check that we don't have a mixed linear/bilinear form or anything like that
    # FIXME: This is slooow and should be moved to form compiler and/or replaced with something faster
    ufl_assert(len(compute_form_arities(preprocessed_form)) == 1,
               "All terms in form must have same rank.")


def compute_form_data(form, apply_propagate_restrictions=True):

    # TODO: Move this to the constructor instead
    self = FormData()

    # Store untouched form for reference.
    # The user of FormData may get original arguments,
    # original coefficients, and form signature from this object.
    # But be aware that the set of original coefficients are not
    # the same as the ones used in the final UFC form.
    # See 'reduced_coefficients' below.
    self.original_form = form

    # Get rank of form from argument list (assuming not a mixed arity form)
    self.rank = len(form.arguments())

    # Extract common geometric dimension (topological is not common!)
    gdims = set(domain.geometric_dimension() for domain in form.domains())
    ufl_assert(len(gdims) == 1,
               "Expecting all integrals in a form to share geometric dimension, got %s." % str(tuple(sorted(gdims))))
    self.geometric_dimension, = gdims

    # Build mapping from old incomplete element objects to new well defined elements.
    # This is to support the Expression construct in dolfin which subclasses Coefficient
    # but doesn't provide an element, and the Constant construct that doesn't provide
    # the domain that a Coefficient is supposed to have. A future design iteration in
    # UFL/UFC/FFC/DOLFIN may allow removal of this mapping with the introduction of UFL
    # types for .
    self.element_replace_map = _compute_element_mapping(form)


    # --- Pass form integrands through some symbolic manipulation

    # Process form the way that is currently expected by FFC
    preprocessed_form = expand_derivatives(form)

    if apply_propagate_restrictions:
        preprocessed_form = propagate_restrictions(preprocessed_form)


    # --- Group and collect data about integrals
    # TODO: Refactor this
    self.integral_data = \
        build_integral_data(preprocessed_form.integrals(), form.domains())


    # --- Create replacements for arguments and coefficients

    # Figure out which form coefficients each integral should enable
    for itg_data in self.integral_data:
        itg_coeffs = set()
        # Get all coefficients in integrand
        for itg in itg_data.integrals:
            itg_coeffs.update(extract_coefficients(itg.integrand()))
        # Add coefficient for integration domain if any
        c = itg_data.domain.coordinates()
        if c is not None:
            itg_coeffs.add(c)
        # Store with IntegralData object
        itg_data.integral_coefficients = itg_coeffs

    # Figure out which coefficients from the original form are actually used in any integral
    # (Differentiation may reduce the set of coefficients w.r.t. the original form)
    reduced_coefficients_set = set()
    for itg_data in self.integral_data:
        reduced_coefficients_set.update(itg_data.integral_coefficients)
    self.reduced_coefficients = sorted(reduced_coefficients_set, key=lambda c: c.count())
    self.num_coefficients = len(self.reduced_coefficients)
    self.original_coefficient_positions = [i for i,c in enumerate(form.coefficients())
                                           if c in self.reduced_coefficients]

    # Store back into integral data which form coefficients are used by each integral
    for itg_data in self.integral_data:
        itg_data.enabled_coefficients = [bool(coeff in itg_data.integral_coefficients)
                                         for coeff in self.reduced_coefficients]

    """
    # Build mappings from coefficients, domains and geometric quantities
    # that reside in form to objects with canonical numbering as well as
    # completed elements

    coordinate_functions = set(domain.coordinates() for domain in form.domains()) - set((None,))

    coordinates_replace_map = {}
    for i, f in enumerate(self.reduced_coefficients):
        if f in coordinate_functions:
            new_f = f.reconstruct(count=i)
            coordinates_replace_map[f] = new_f

    domains_replace_map = {}
    for domain in form.domains():
        FIXME

    geometry_replace_map = {}
    FIXME

    coefficients_replace_map = {}
    for i, f in enumerate(self.reduced_coefficients):
        if f not in coordinate_functions:
            old_e = f.element()
            new_e = self.element_replace_map.get(old_e, old_e)
            new_f = f.reconstruct(element=new_e, count=i)
            coefficients_replace_map[f] = new_f

    self.terminals_replace_map = {}
    self.terminals_replace_map.update(coordinates_replace_map)
    self.terminals_replace_map.update(domains_replace_map) # Not currently terminals but soon will be
    self.terminals_replace_map.update(geometry_replace_map)
    self.terminals_replace_map.update(coefficients_replace_map)

    renumbered_coefficients = [self.terminals_replace_map[f] for f in self.reduced_coefficients]
    """

    # Mappings from elements and coefficients
    # that reside in form to objects with canonical numbering as well as
    # completed cells and elements
    renumbered_coefficients, function_replace_map = \
        build_coefficient_replace_map(self.reduced_coefficients, self.element_replace_map)
    self.function_replace_map = function_replace_map

    # --- Store various lists of elements and sub elements (adds members to self)
    _compute_form_data_elements(self, form.arguments(), renumbered_coefficients)

    # --- Store number of domains for integral types
    # TODO: Group this by domain first. For now keep a backwards compatible data structure.
    self.num_sub_domains = _compute_num_sub_domains(self.integral_data)


    # --- Checks
    _check_elements(self)
    _check_facet_geometry(self.integral_data)

    # TODO: This is a very expensive check... Replace with something faster!
    preprocessed_form = reconstruct_form_from_integral_data(self.integral_data)
    _check_form_arity(preprocessed_form)

    # TODO: This is used by unit tests, change the tests!
    self.preprocessed_form = preprocessed_form

    return self
