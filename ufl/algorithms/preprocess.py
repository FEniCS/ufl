"""This module provides the preprocess function which form compilers
will typically call prior to code generation to preprocess/simplify a
raw input form given by a user."""

# Copyright (C) 2009-2013 Anders Logg and Martin Sandve Alnes
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
# First added:  2009-12-07
# Last changed: 2012-04-12

from itertools import chain
from time import time
import ufl
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.form import Form
from ufl.common import slice_dict
from ufl.geometry import Cell
from ufl.domains import as_domain
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.replace import replace
from ufl.algorithms.analysis import (extract_arguments_and_coefficients,
                                     build_argument_replace_map,
                                     extract_elements, extract_sub_elements,
                                     unique_tuple, _domain_types,
                                     extract_num_sub_domains, extract_domain_data)
from ufl.algorithms.domain_analysis import (
    integral_dict_to_sub_integral_data,
    print_sub_integral_data,
    reconstruct_form_from_sub_integral_data,
    convert_sub_integral_data_to_integral_data)
from ufl.algorithms.formdata import FormData, ExprData
from ufl.algorithms.expand_indices import expand_indices
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.propagate_restrictions import propagate_restrictions
from ufl.algorithms.formtransformations import compute_form_arities
from ufl.algorithms.signature import compute_expression_signature, compute_form_signature

class Timer:
    def __init__(self, name):
        self.name = name
        self.times = []
        self('begin %s' % self.name)
    def __call__(self, msg):
        self.times.append((time(), msg))
    def end(self):
        self('end %s' % self.name)
    def __str__(self):
        line = "-"*60
        s = [line, "Timing of %s" % self.name]
        for i in range(len(self.times)-1):
            t = self.times[i+1][0] - self.times[i][0]
            msg = self.times[i][1]
            s.append("%9.2e s    %s" % (t, msg))
        s.append('Total time: %9.2e s' % (self.times[-1][0] - self.times[0][0]))
        s.append(line)
        return '\n'.join(s)

def preprocess(form, object_names=None, common_cell=None, element_mapping=None):
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

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting Form.")
    original_form = form

    # Object names is empty if not given
    object_names = object_names or {}

    # Element mapping is empty if not given
    element_mapping = element_mapping or {}

    # Create empty form data
    form_data = FormData()

    # Store copies of preprocess input data, for future validation if called again...
    form_data._input_object_names = dict(object_names)
    form_data._input_element_mapping = dict(element_mapping)
    #form_data._input_common_cell = no need to store this

    # Store name of form if given, otherwise empty string
    # such that automatic names can be assigned externally
    form_data.name = object_names.get(id(form), "")

    # Extract common cell
    common_cell = extract_common_cell(form, common_cell)

    # TODO: Split out expand_compounds from expand_derivatives
    # Expand derivatives
    tic('expand_derivatives')
    form = expand_derivatives(form, common_cell.geometric_dimension()) # EXPR

    # Propagate restrictions of interior facet integrals to the terminal nodes
    form = propagate_restrictions(form) # INTEGRAL, EXPR

    # --- BEGIN DOMAIN SPLITTING AND JOINING
    # Store domain metadata per domain type
    form_data.domain_data = extract_domain_data(form)

    # Split up integrals and group by domain type and domain id,
    # adding integrands with same domain and compiler data
    sub_integral_data = integral_dict_to_sub_integral_data(form.integral_groups())
    # TODO: Replace integral_data with this through ufl and ffc?
    if 0: print_sub_integral_data(sub_integral_data)

    # Reconstruct form from these integrals in a more canonical representation
    form = reconstruct_form_from_sub_integral_data(sub_integral_data, form_data.domain_data)

    # Represent in the way ffc expects TODO: Change ffc? Fine for now.
    form_data.integral_data = convert_sub_integral_data_to_integral_data(sub_integral_data)
    # --- END DOMAIN SPLITTING AND JOINING


    # --- BEGIN FUNCTION ANALYSIS
    # --- BEGIN SPLIT EXPR JOIN
    # Replace arguments and coefficients with new renumbered objects
    tic('extract_arguments_and_coefficients')
    original_arguments, original_coefficients = \
        extract_arguments_and_coefficients(form)

    tic('build_element_mapping')
    element_mapping = build_element_mapping(element_mapping,
                                            common_cell,
                                            original_arguments,
                                            original_coefficients)

    tic('build_argument_replace_map') # TODO: Remove renumbered ones?
    replace_map, renumbered_arguments, renumbered_coefficients = \
        build_argument_replace_map(original_arguments,
                                   original_coefficients,
                                   element_mapping)

    # Note: This is the earliest point signature can be computed

    # Build mapping to original arguments and coefficients, which is
    # useful if the original arguments have data attached to them
    inv_replace_map = dict((w,v) for (v,w) in replace_map.iteritems())
    original_arguments = [inv_replace_map[v] for v in renumbered_arguments]
    original_coefficients = [inv_replace_map[w] for w in renumbered_coefficients]

    # TODO: Build mapping from object to position instead? But we need mapped elements as well anyway.
    #argument_positions = { v: i }
    #coefficient_positions = { w: i }

    # Store data extracted by preprocessing
    form_data.original_arguments      = original_arguments
    form_data.original_coefficients   = original_coefficients
    # --- END SPLIT INTEGRAL JOIN

    # Mappings from elements and functions (coefficients and arguments)
    # that reside in form to objects with canonical numbering as well as
    # completed cells and elements
    # TODO: Create additional function mappings per integral,
    #       to have different counts? Depends on future UFC design.
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
    # --- END FUNCTION ANALYSIS


    # --- BEGIN SIGNATURE COMPUTATION
    # TODO: Compute signatures of each INTEGRAL and EXPR as well, perhaps compute it hierarchially from integral_data?
    # Store signature of form
    tic('signature')
    # TODO: Remove signature() from Form, not safe to cache with a replacement map
    #form_data.signature = form.signature(form_data.function_replace_map)
    form_data.signature = compute_form_signature(form, form_data.function_replace_map)
    # --- END SIGNATURE COMPUTATION


    # --- BEGIN CONSISTENCY CHECKS
    # Check that we don't have a mixed linear/bilinear form or anything like that
    ufl_assert(len(compute_form_arities(form)) == 1, "All terms in form must have same rank.")
    # --- END CONSISTENCY CHECKS


    # --- BEGIN ELEMENT DATA
    # Store elements, sub elements and element map
    tic('extract_elements')
    form_data.argument_elements    = tuple(f.element() for f in renumbered_arguments)
    form_data.coefficient_elements = tuple(f.element() for f in renumbered_coefficients)
    form_data.elements             = form_data.argument_elements + form_data.coefficient_elements
    form_data.unique_elements      = unique_tuple(form_data.elements)
    form_data.sub_elements         = extract_sub_elements(form_data.elements)
    form_data.unique_sub_elements  = unique_tuple(form_data.sub_elements)
    # --- END ELEMENT DATA


    # --- BEGIN DOMAIN DATA
    # Store element domains (NB! This is likely to change!)
    # TODO: DOMAINS: What is a sensible way to store domains for a form?
    form_data.domains = tuple(sorted(set(element.domain()
                                         for element in form_data.unique_elements)))

    # Store toplevel domains (NB! This is likely to change!)
    # TODO: DOMAINS: What is a sensible way to store domains for a form?
    form_data.top_domains = tuple(sorted(set(domain.top_domain()
                                             for domain in form_data.domains)))

    # Store common cell
    form_data.cell = common_cell

    # Store data related to cell
    form_data.geometric_dimension = form_data.cell.geometric_dimension()
    form_data.topological_dimension = form_data.cell.topological_dimension()

    # Store number of domains for integral types
    form_data.num_sub_domains = extract_num_sub_domains(form)
    # --- END DOMAIN DATA


    # A coarse profiling implementation TODO: Add counting of nodes, Add memory usage
    tic.end()
    if preprocess.enable_profiling:
        print tic

    # Store preprocessed form and return
    form_data.preprocessed_form = form
    form_data.preprocessed_form._is_preprocessed = True

    # Attach signatures to original and preprocessed forms TODO: Avoid this?
    ufl_assert(form_data.preprocessed_form._signature is None, "")
    form_data.preprocessed_form._signature = form_data.signature
    ufl_assert(original_form._signature is None, "")
    original_form._signature = form_data.signature

    return form_data
preprocess.enable_profiling = False

def preprocess_expression(expr, object_names=None, common_cell=None, element_mapping=None):
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
    tic = Timer('preprocess_expression')

    # Check that we get an expression
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")

    # Object names is empty if not given
    object_names = object_names or {}

    # Element mapping is empty if not given
    element_mapping = element_mapping or {}

    # Create empty expression data
    expr_data = ExprData()

    # Store original expression
    expr_data.original_expr = expr

    # Store name of expr if given, otherwise empty string
    # such that automatic names can be assigned externally
    expr_data.name = object_names.get(id(expr), "") # TODO: Or default to 'expr'?

    # Extract common cell
    common_cell = extract_common_cell(expr, common_cell)

    # TODO: Split out expand_compounds from expand_derivatives
    # Expand derivatives
    tic('expand_derivatives')
    expr = expand_derivatives(expr, common_cell.geometric_dimension())

    # Renumber indices
    #expr = renumber_indices(expr) # TODO: No longer needed?

    # Replace arguments and coefficients with new renumbered objects
    tic('extract_arguments_and_coefficients')
    original_arguments, original_coefficients = \
        extract_arguments_and_coefficients(expr)

    tic('build_element_mapping')
    element_mapping = build_element_mapping(element_mapping,
                                            common_cell,
                                            original_arguments,
                                            original_coefficients)

    tic('build_argument_replace_map')
    replace_map, renumbered_arguments, renumbered_coefficients = \
        build_argument_replace_map(original_arguments,
                                   original_coefficients,
                                   element_mapping)
    expr = replace(expr, replace_map)

    # Build mapping to original arguments and coefficients, which is
    # useful if the original arguments have data attached to them
    inv_replace_map = dict((w,v) for (v,w) in replace_map.iteritems())
    original_arguments = [inv_replace_map[v] for v in renumbered_arguments]
    original_coefficients = [inv_replace_map[w] for w in renumbered_coefficients]

    # Store data extracted by preprocessing
    expr_data.arguments               = renumbered_arguments    # TODO: Needed?
    expr_data.coefficients            = renumbered_coefficients # TODO: Needed?
    expr_data.original_arguments      = original_arguments
    expr_data.original_coefficients   = original_coefficients
    expr_data.renumbered_arguments    = renumbered_arguments
    expr_data.renumbered_coefficients = renumbered_coefficients

    tic('replace')
    # Mappings from elements and functions (coefficients and arguments)
    # that reside in expr to objects with canonical numbering as well as
    # completed cells and elements
    expr_data.element_replace_map = element_mapping
    expr_data.function_replace_map = replace_map

    # Store signature of form
    tic('signature')
    expr_data.signature = compute_expression_signature(expr, expr_data.function_replace_map)

    # Store elements, sub elements and element map
    tic('extract_elements')
    expr_data.elements            = tuple(f.element() for f in
                                          chain(renumbered_arguments,
                                                renumbered_coefficients))
    expr_data.unique_elements     = unique_tuple(expr_data.elements)
    expr_data.sub_elements        = extract_sub_elements(expr_data.elements)
    expr_data.unique_sub_elements = unique_tuple(expr_data.sub_elements)

    # Store element domains (NB! This is likely to change!)
    # FIXME: DOMAINS: What is a sensible way to store domains for a expr?
    expr_data.domains = tuple(sorted(set(element.domain()
                                         for element in expr_data.unique_elements)))

    # Store toplevel domains (NB! This is likely to change!)
    # FIXME: DOMAINS: What is a sensible way to store domains for a expr?
    expr_data.top_domains = tuple(sorted(set(domain.top_domain()
                                             for domain in expr_data.domains)))

    # Store common cell
    expr_data.cell = common_cell

    # Store data related to cell
    expr_data.geometric_dimension = expr_data.cell.geometric_dimension()
    expr_data.topological_dimension = expr_data.cell.topological_dimension()

    # Store some useful dimensions
    expr_data.rank = len(expr_data.arguments) # TODO: Is this useful for expr?
    expr_data.num_coefficients = len(expr_data.coefficients)

    # Store argument names # TODO: Is this useful for expr?
    expr_data.argument_names = \
        [object_names.get(id(expr_data.original_arguments[i]), "v%d" % i)
         for i in range(expr_data.rank)]

    # Store coefficient names
    expr_data.coefficient_names = \
        [object_names.get(id(expr_data.original_coefficients[i]), "w%d" % i)
         for i in range(expr_data.num_coefficients)]

    # Store preprocessed expression
    expr_data.preprocessed_expr = expr

    tic.end()

    # A coarse profiling implementation
    # TODO: Add counting of nodes
    # TODO: Add memory usage
    if preprocess_expression.enable_profiling:
        print tic

    return expr_data
preprocess_expression.enable_profiling = False

def extract_common_cell(form, common_cell=None):
    "Extract common cell for form or expression."

    # Either use given argument or try to find in form or expression
    common_cell = common_cell or form.cell()

    # Check common cell
    if common_cell is None:
        error("Unable to extract common cell; "\
              "missing cell definition in form or expression.")

    return common_cell

def build_element_mapping(element_mapping, common_cell, arguments, coefficients):
    """Complete an element mapping for all elements used by
    arguments and coefficients, using a well defined common cell."""

    # Build a new dict to avoid modifying the dict passed from non-ufl code
    new_element_mapping = {}

    # Check that the given initial mapping has no invalid entries as values
    for v in element_mapping.itervalues():
        ufl_assert(v.cell() is not None,
                   "Found incomplete element with undefined cell in element mapping.")
        ufl_assert(v.family() is not None,
                   "Found incomplete element with undefined family in element mapping.")

    common_domain = as_domain(common_cell) # FIXME: handle better somehow?

    # Reconstruct all elements we need to map
    for f in chain(arguments, coefficients):
        e = f.element()
        # Prefer the given mapping:
        new_e = element_mapping.get(e)
        if new_e is None:
            if e.cell() is None:
                # Otherwise complete with domain by reconstructing if cell is missing
                new_e = e.reconstruct(domain=common_domain)
            else:
                # Or just use the original element
                new_e = e
        new_element_mapping[e] = new_e

    # Check that the new mapping has no invalid entries as values
    for v in new_element_mapping.itervalues():
        ufl_assert(v.cell() is not None,
                   "Found incomplete element with undefined cell in new element mapping.")
        ufl_assert(v.family() is not None,
                   "Found incomplete element with undefined family in new element mapping.")

    return new_element_mapping
