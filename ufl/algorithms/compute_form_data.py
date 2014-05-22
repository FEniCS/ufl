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
                                     unique_tuple,
                                     extract_num_sub_domains)
from ufl.algorithms.domain_analysis import build_integral_data, reconstruct_form_from_integral_data
from ufl.algorithms.formdata import FormData, ExprData
from ufl.algorithms.expand_indices import expand_indices
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.propagate_restrictions import propagate_restrictions
from ufl.algorithms.formtransformations import compute_form_arities
from ufl.algorithms.signature import compute_expression_signature, compute_form_signature
from ufl.algorithms.preprocess import _compute_element_mapping, build_element_mapping


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

def _compute_element_mapping(elements, common_domain):
    "Compute element mapping for element replacement"

    # Try to find a common degree for elements
    elements = extract_sub_elements(elements)
    common_degree = _auto_select_degree(elements)

    # Compute element map
    element_mapping = {}
    for element in elements:

        # Flag for whether element needs to be reconstructed
        reconstruct = False

        # Set domain/cell
        domain = element.domain()
        if domain is None:
            ufl_assert(common_domain is not None,
                       "Cannot replace unknown element domain without unique common domain in form.")
            info("Adjusting missing element domain to %s." % (common_domain,))
            domain = common_domain
            reconstruct = True

        # Set degree
        degree = element.degree()
        if degree is None:
            info("Adjusting missing element degree to %d" % (common_degree,))
            degree = common_degree
            reconstruct = True

        # Reconstruct element and add to map
        if reconstruct:
            element_mapping[element] = element.reconstruct(domain=domain,
                                                           degree=degree)

    return element_mapping

def build_element_mapping(element_mapping, common_domain, arguments, coefficients):
    """Complete an element mapping for all elements used by
    arguments and coefficients, using a well defined common domain."""

    # Build a new dict to avoid modifying the dict passed from non-ufl code
    new_element_mapping = {}

    # Check that the given initial mapping has no invalid entries as values
    for element in element_mapping.itervalues():
        ufl_assert(element.domain() is not None,
                   "Found incomplete element with undefined domain in element mapping.")
        ufl_assert(element.family() is not None,
                   "Found incomplete element with undefined family in element mapping.")

    # Reconstruct all elements we need to map
    for f in chain(arguments, coefficients):
        element = f.element()
        # Prefer the given mapping:
        new_element = element_mapping.get(element)
        if new_element is None:
            if element.domain() is None:
                # Otherwise complete with domain by reconstructing if domain is missing
                new_element = element.reconstruct(domain=common_domain)
            else:
                # Or just use the original element
                new_element = element
        new_element_mapping[element] = new_element

    # Check that the new mapping has no invalid entries as values
    for element in new_element_mapping.itervalues():
        ufl_assert(element.domain() is not None,
                   "Found incomplete element with undefined domain in new element mapping.")
        ufl_assert(element.family() is not None,
                   "Found incomplete element with undefined family in new element mapping.")

    return new_element_mapping

def compute_form_data(form, object_names):
    self = FormData()

    # Check input
    ufl_assert(isinstance(form, Form), "Expecting Form.")
    object_names = object_names or {}

    # Extract form arguments
    self.original_arguments = form.arguments()
    self.original_coefficients = form.coefficients()

    self.rank = len(self.original_arguments)
    #self.original_num_coefficients = len(self.original_coefficients)

    # Store name of form if given, otherwise empty string
    # such that automatic names can be assigned externally
    self.name = object_names.get(id(form), "")

    # Extract common domain
    self.domains = form.domains()
    common_domain = self.domains[0] if len(self.domains) == 1 else None

    # Compute signature, this can be a bit costly
    self.signature = form.signature()


    # --- Pass form through some symbolic manipulation

    # FIXME: Extract this part such that a different symbolic pipeline can be used for uflacs.

    # Process form the way that is currently expected by FFC
    preprocessed_form = expand_derivatives(form)
    preprocessed_form = propagate_restrictions(preprocessed_form)

    # Build list of integral data objects (also does quite a bit of processing)
    # TODO: This is unclear, explain what kind of processing and/or refactor
    self.integral_data = \
        build_integral_data(preprocessed_form.integrals(), self.domains, common_domain)

    # Reconstruct final preprocessed form from these integrals,
    # in a more canonical representation than the original input
    self.preprocessed_form = reconstruct_form_from_integral_data(self.integral_data)


    # --- Create replacements for arguments and coefficients

    # Build mapping from old incomplete element objects to new well defined elements
    # TODO: Refactor: merge build_element_mapping and _compute_element_mapping
    element_mapping = _compute_element_mapping(extract_elements(form), common_domain)
    element_mapping = build_element_mapping(element_mapping,
                                            common_domain,
                                            self.original_arguments,
                                            self.original_coefficients)

    # Figure out which form coefficients each integral should enable
    reduced_coefficients_set = set()
    for itg_data in self.integral_data:
        itg_coeffs = set()
        for itg in itg_data.integrals:
            itg_coeffs.update(extract_coefficients(itg.integrand()))
        itg_data.integral_coefficients = itg_coeffs
        reduced_coefficients_set.update(itg_coeffs)

    self.reduced_coefficients = sorted(reduced_coefficients_set, key=lambda c: c.count())
    self.num_coefficients = len(self.reduced_coefficients)
    for itg_data in self.integral_data:
        itg_data.enabled_coefficients = [bool(coeff in itg_data.integral_coefficients)
                                         for coeff in self.reduced_coefficients]

    self.original_coefficient_positions = [i for i,c in enumerate(self.original_coefficients)
                                           if c in self.reduced_coefficients]

    renumbered_coefficients, replace_map = \
        build_coefficient_replace_map(self.reduced_coefficients, element_mapping)

    # Mappings from elements and coefficients
    # that reside in form to objects with canonical numbering as well as
    # completed cells and elements
    self.element_replace_map = element_mapping
    self.function_replace_map = replace_map


    # --- Store elements, sub elements and element map
    self.argument_elements    = tuple(f.element() for f in self.original_arguments)
    self.coefficient_elements = tuple(f.element() for f in renumbered_coefficients)
    self.elements             = self.argument_elements + self.coefficient_elements
    self.unique_elements      = unique_tuple(self.elements)
    self.sub_elements         = extract_sub_elements(self.elements)
    self.unique_sub_elements  = unique_tuple(self.sub_elements)


    # --- Store geometry data
    self.integration_domains = self.preprocessed_form.domains()
    if self.integration_domains:
        self.geometric_dimension = self.integration_domains[0].geometric_dimension()
    else:
        warning("Got no integration domains!")


    # --- Store number of domains for integral types
    self.num_sub_domains = extract_num_sub_domains(self.preprocessed_form)

    # TODO: Support multiple domains throughout jit chain. For now keep a backwards compatible data structure.
    ufl_assert(len(self.num_sub_domains) == 1, "Not used for multiple domains yet. Might work.")
    self.num_sub_domains, = self.num_sub_domains.values()


    # Store argument names
    self.argument_names = \
        [object_names.get(id(self.original_arguments[i]), "v%d" % i)
        for i in range(self.rank)]

    # Store coefficient names
    self.coefficient_names = \
        [object_names.get(id(self.reduced_coefficients[i]), "w%d" % i)
        for i in range(self.num_coefficients)]


    # --- Checks

    for itg_data in self.integral_data:
        for itg in itg_data.integrals:
            classes = extract_classes(itg.integrand())
            it = itg_data.integral_type
            # Facet geometry is only valid in facet integrals
            if "facet" not in it:
                for c in classes:
                    ufl_assert(not issubclass(c, GeometricFacetQuantity),
                               "Integral of type %s cannot contain a %s." % (it, c.__name__))

    # Check that we don't have a mixed linear/bilinear form or anything like that
    # FIXME: This is slooow and should be moved to form compiler and/or replaced with something faster
    ufl_assert(len(compute_form_arities(self.preprocessed_form)) == 1,
               "All terms in form must have same rank.")

    return self
