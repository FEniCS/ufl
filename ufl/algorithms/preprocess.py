"""This module provides the preprocess function which form compilers
will typically call prior to code generation to preprocess/simplify a
raw input form given by a user."""

# Copyright (C) 2009-2014 Anders Logg and Martin Sandve Alnes
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

from itertools import chain
import six
from time import time
import ufl
from ufl.log import error, warning, info
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.form import Form
from ufl.common import istr, slice_dict
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
from ufl.common import Timer

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

from collections import defaultdict

def join_subdomain_data(integrals, domains):
    labels = set((domain.label() if domain is not None else None) for domain in domains)

    l2l = dict((l,l) for l in labels)
    if None in labels:
        labels.remove(None)
        if len(labels) == 1:
            l, = labels
            l2l[None] = l
        elif len(labels) == 0:
            l2l[None] = None
        else:
            error("With None label, only one other label can exist, found %s." % (labels,))

    # Collect subdomain data objects for each domain
    subdomain_data = defaultdict(dict)
    for itg in integrals:
        # Get data and skip if there's none
        dd = itg.subdomain_data()
        if dd is None:
            continue

        # Get a domain label to store data with
        # This is messy, not sure if we need to make it this complicated?
        d = itg.domain()
        if d is None:
            l = l2l[None]
        else:
            l = l2l[d.label()]
        dt = itg.integral_type()

        # Store data with label/domain type, or make sure it'd compatible if already stored
        old_dd = subdomain_data[l].get(dt)

        if old_dd is None:
            subdomain_data[l][dt] = dd
        elif id_or_none(old_dd) != id_or_none(dd):
            error("Subdomain data object mismatch in form, for label %s and domain type %s." % (l,dt))

    return subdomain_data

def preprocess(form, object_names=None):
    """
    Preprocess raw input form to obtain form metadata, including a
    modified (preprocessed) form more easily manipulated by form
    compilers. The original form is left untouched. Currently, the
    following transformations are made to the preprocessed form:

      expand_compounds    (side effect of calling expand_derivatives)
      expand_derivatives
      renumber arguments and coefficients and apply evt. element mapping
    """
    tic = Timer('preprocess') # TODO: Reposition tic calls after refactoring.

    # Create empty form data
    form_data = FormData()

    # --- Arguments

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting Form.")
    original_form = form

    # Object names is empty if not given
    object_names = object_names or {}

    # Store copies of preprocess input data, for future validation if called again...
    form_data._input_object_names = dict(object_names)

    # Store name of form if given, otherwise empty string
    # such that automatic names can be assigned externally
    form_data.name = object_names.get(id(original_form), "")

    # --- Processing form

    # Store collection of subdomain data objects for each domain label x domain type
    tic('join_subdomain_data')
    form_data.subdomain_data = join_subdomain_data(form.integrals(), form.domains())

    # Propagate derivatives to the terminal nodes
    tic('expand_derivatives')
    # TODO: Split out expand_compounds from expand_derivatives
    form = expand_derivatives(original_form)

    # Propagate restrictions of interior facet integrals to the terminal nodes
    tic('propagate_restrictions')
    form = propagate_restrictions(form)

    # Extract common domain and build basic element mapping
    domains = form.domains()
    common_domain = domains[0] if len(domains) == 1 else None

    # Build list of integral data objects (also does quite a bit of processing)
    # TODO: This is unclear, explain what kind of processing and/or refactor
    tic('build_integral_data')
    form_data.integral_data = \
        build_integral_data(form.integrals(), domains, common_domain)

    # Reconstruct final preprocessed form from these integrals,
    # in a more canonical representation than the original input
    tic('reconstruct_form_from_integral_data')
    form = reconstruct_form_from_integral_data(form_data.integral_data)

    # Store final preprocessed form
    form_data.preprocessed_form = form

    # --- Create replacements for arguments and coefficients

    # Fetch non-renumbered argument and coefficient objects
    tic('extract_arguments_and_coefficients')
    original_arguments, original_coefficients = \
        extract_arguments_and_coefficients(form_data.preprocessed_form)

    # Build mapping from old incomplete element objects to new well defined elements
    tic('build_element_mapping')
    # TODO: Refactor: merge build_element_mapping and _compute_element_mapping
    element_mapping = _compute_element_mapping(extract_elements(original_form), common_domain)
    element_mapping = build_element_mapping(element_mapping,
                                            common_domain,
                                            original_arguments,
                                            original_coefficients)

    tic('build_coefficient_replace_map')
    renumbered_coefficients, replace_map = \
        build_coefficient_replace_map(original_coefficients, element_mapping)

    tic('build enabled_coefficients lists')
    for itg_data in form_data.integral_data:
        itg_coeffs = set()
        for itg in itg_data.integrals:
            itg_coeffs.update(extract_coefficients(itg.integrand()))
        itg_data.enabled_coefficients = [bool(coeff in itg_coeffs) for coeff in original_coefficients]

    #############################################################################
    # Note: This is the earliest point the final form signature can be computed,
    #       because it depends on the renumbering and element mapping
    #############################################################################

    # Build mapping to original coefficients, which is
    # useful if the original coefficient have data attached to them
    inv_replace_map = dict((w,v) for (v,w) in six.iteritems(replace_map))

    # TODO: What's the point of this? Added assertion to check for sanity.
    original_coefficients2 = [inv_replace_map[w] for w in renumbered_coefficients]
    ufl_assert(all(c1 == c2 for c1,c2 in zip(original_coefficients, original_coefficients2)),
               "Got two versions of original coefficients?")

    # TODO: Build mapping from object to position instead? But we need mapped elements as well anyway.
    #       For Arguments, the plan now is to change count to number all over, which removes the
    #       need for renumbering of Arguments.
    #       For Coefficients, a suggestion has been to allow a different local numbering per integral,
    #       which lends itself nicely to storing coefficient_positions for the form and the integrals
    #       separately instead of storing the function replace map.
    #argument_positions = { v: i }
    #coefficient_positions = { w: i }

    # Store data extracted by preprocessing
    form_data.original_arguments      = original_arguments
    form_data.original_coefficients   = original_coefficients

    # Mappings from elements and coefficients
    # that reside in form to objects with canonical numbering as well as
    # completed cells and elements
    form_data.element_replace_map = element_mapping
    form_data.function_replace_map = replace_map

    # Store some useful dimensions
    form_data.rank = len(form_data.original_arguments)
    form_data.num_coefficients = len(form_data.original_coefficients)

    # Store argument names
    form_data.argument_names = \
        [object_names.get(id(form_data.original_arguments[i]), "v%d" % i)
         for i in range(form_data.rank)]

    # Store coefficient names
    form_data.coefficient_names = \
        [object_names.get(id(form_data.original_coefficients[i]), "w%d" % i)
         for i in range(form_data.num_coefficients)]


    # Store signature of form
    tic('signature')
    # TODO: Remove signature() from Form, not safe to cache with a replacement map (TODO: Is it safe now?)
    #form_data.signature = form.signature(form_data.function_replace_map)
    form_data.signature = compute_form_signature(form_data.preprocessed_form,
                                                 form_data.function_replace_map)

    # TODO: Compute signatures of each INTEGRAL and EXPR as well,
    #       perhaps compute it hierarchially from integral_data?


    # --- Checks
    tic('error checking')
    for itg_data in form_data.integral_data:
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
    ufl_assert(len(compute_form_arities(form_data.preprocessed_form)) == 1,
               "All terms in form must have same rank.")


    # --- Elements

    # Store elements, sub elements and element map
    tic('extract_elements')
    form_data.argument_elements    = tuple(f.element() for f in original_arguments)
    form_data.coefficient_elements = tuple(f.element() for f in renumbered_coefficients)
    form_data.elements             = form_data.argument_elements + form_data.coefficient_elements
    form_data.unique_elements      = unique_tuple(form_data.elements)
    form_data.sub_elements         = extract_sub_elements(form_data.elements)
    form_data.unique_sub_elements  = unique_tuple(form_data.sub_elements)


    # --- Geometry

    # Store geometry data
    form_data.integration_domains = form_data.preprocessed_form.domains()
    if form_data.integration_domains:
        form_data.geometric_dimension = form_data.integration_domains[0].geometric_dimension()
    else:
        warning("Got no integration domains!")

    # Store number of domains for integral types
    form_data.num_sub_domains = extract_num_sub_domains(form_data.preprocessed_form)

    # TODO: Support multiple domains throughout jit chain. For now keep a backwards compatible data structure.
    ufl_assert(len(form_data.num_sub_domains) == 1, "Not used for multiple domains yet. Might work.")
    form_data.num_sub_domains, = list(form_data.num_sub_domains.values())


    # --- Caching

    # Attach signatures to original and preprocessed forms TODO: Avoid this?
    ufl_assert(form_data.preprocessed_form._signature is None, "")
    ufl_assert(original_form._signature is None, "")
    form_data.preprocessed_form._is_preprocessed = True
    form_data.preprocessed_form._signature = form_data.signature
    original_form._signature = form_data.signature


    # A coarse profiling implementation TODO: Add counting of nodes, Add memory usage
    tic.end()
    if preprocess.enable_profiling:
        print(tic)

    return form_data
preprocess.enable_profiling = False


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
    for element in six.itervalues(element_mapping):
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
    for element in six.itervalues(new_element_mapping):
        ufl_assert(element.domain() is not None,
                   "Found incomplete element with undefined domain in new element mapping.")
        ufl_assert(element.family() is not None,
                   "Found incomplete element with undefined family in new element mapping.")

    return new_element_mapping

# FIXME: Remove this and just reuse preprocess_form with expr*dirac_delta_measure?
def preprocess_expression(expr, object_names=None):
    """
    Preprocess raw input expression to obtain expression metadata,
    including a modified (preprocessed) expression more easily
    manipulated by expression compilers. The original expression
    is left untouched. Currently, the following transformations
    are made to the preprocessed form:

      expand_compounds    (side effect of calling expand_derivatives)
      expand_derivatives
      renumber arguments and coefficients and apply evt. element mapping
    """
    tic = Timer('preprocess_expression') # TODO: Reposition tic calls after refactoring.

    # Create empty expression data
    expr_data = ExprData()

    # --- Arguments

    # Check that we get an expression
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")
    original_expr = expr

    # Object names is empty if not given
    object_names = object_names or {}

    # Store copies of preprocess input data, for future validation if called again...
    expr_data._input_object_names = dict(object_names)

    # Store name of expr if given, otherwise empty string
    # such that automatic names can be assigned externally
    expr_data.name = object_names.get(id(expr), "") # TODO: Or default to 'expr'?


    # --- Processing expression

    # Expand derivatives
    tic('expand_derivatives')
    # TODO: Split out expand_compounds from expand_derivatives
    expr = expand_derivatives(expr)

    # Extract common domain
    domains = expr.domains()
    common_domain = domains[0] if len(domains) == 1 else None

    # Store preprocessed expression
    expr_data.preprocessed_expr = expr

    # --- Create replacements for arguments and coefficients

    # Replace arguments and coefficients with new renumbered objects
    tic('extract_arguments_and_coefficients')
    original_arguments, original_coefficients = \
        extract_arguments_and_coefficients(expr_data.preprocessed_expr)

    # Build mapping from old incomplete element objects to new well defined elements
    tic('build_element_mapping')
    # TODO: Refactor: merge build_element_mapping and _compute_element_mapping
    element_mapping = _compute_element_mapping(extract_elements(original_expr), common_domain)
    element_mapping = build_element_mapping(element_mapping,
                                            common_domain,
                                            original_arguments,
                                            original_coefficients)

    tic('build_coefficient_replace_map')
    renumbered_coefficients, replace_map = \
        build_coefficient_replace_map(original_coefficients, element_mapping)

    #############################################################################
    # Note: This is the earliest point the final expr signature can be computed,
    #       because it depends on the renumbering and element mapping
    #############################################################################

    # Build mapping to original arguments and coefficients, which is
    # useful if the original arguments have data attached to them
    inv_replace_map = dict((w,v) for (v,w) in six.iteritems(replace_map))

    # TODO: What's the point of this? Added assertion to check for sanity.
    original_coefficients2 = [inv_replace_map[w] for w in renumbered_coefficients]
    ufl_assert(all(c1 == c2 for c1,c2 in zip(original_coefficients, original_coefficients2)),
               "Got two versions of original coefficients?")

    # Store data extracted by preprocessing
    expr_data.original_arguments      = original_arguments
    expr_data.original_coefficients   = original_coefficients

    # Mappings from elements and functions (coefficients and arguments)
    # that reside in expr to objects with canonical numbering as well as
    # completed cells and elements
    expr_data.element_replace_map = element_mapping
    expr_data.function_replace_map = replace_map

    # Store some useful dimensions
    expr_data.rank = len(expr_data.original_arguments)
    expr_data.num_coefficients = len(expr_data.original_coefficients)

    # Store argument names
    expr_data.argument_names = \
        [object_names.get(id(expr_data.original_arguments[i]), "v%d" % i)
         for i in range(expr_data.rank)]

    # Store coefficient names
    expr_data.coefficient_names = \
        [object_names.get(id(expr_data.original_coefficients[i]), "w%d" % i)
         for i in range(expr_data.num_coefficients)]


    # Store signature of expr
    tic('signature')
    expr_data.signature = compute_expression_signature(expr_data.preprocessed_expr,
                                                       expr_data.function_replace_map)

    # --- Checks


    # --- Elements

    # Store elements, sub elements and element map
    tic('extract_elements')
    expr_data.argument_elements    = tuple(f.element() for f in original_arguments)
    expr_data.coefficient_elements = tuple(f.element() for f in renumbered_coefficients)
    expr_data.elements             = expr_data.argument_elements + expr_data.coefficient_elements
    expr_data.unique_elements     = unique_tuple(expr_data.elements)
    expr_data.sub_elements        = extract_sub_elements(expr_data.elements)
    expr_data.unique_sub_elements = unique_tuple(expr_data.sub_elements)


    # --- Geometry

    expr_data.domains = expr_data.preprocessed_expr.domains()
    if expr_data.domains:
        expr_data.geometric_dimension = expr_data.domains[0].geometric_dimension()
    else:
        # TODO: For expressions, this is not solvable (1.0 will not have a domain), but if we do
        #       expr*dP and use form preprocessing we can use expr*dP(domain) to represent this case.
        warning("Got no domains! Geometric dimension is undefined!")

    # A coarse profiling implementation TODO: Add counting of nodes, Add memory usage
    tic.end()
    if preprocess_expression.enable_profiling:
        print(tic)

    return expr_data
preprocess_expression.enable_profiling = False
