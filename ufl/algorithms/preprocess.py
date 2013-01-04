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
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.replace import replace
from ufl.algorithms.analysis import (extract_arguments_and_coefficients,
                                     build_argument_replace_map,
                                     extract_elements, extract_sub_elements,
                                     unique_tuple, _domain_types,
                                     extract_num_sub_domains, extract_domain_data,
                                     extract_integral_data)
from ufl.algorithms.formdata import FormData
from ufl.algorithms.expand_indices import expand_indices

def preprocess(form, object_names=None, common_cell=None, element_mapping=None,
               replace_functions=True, skip_signature=False):
    """
    Preprocess raw input form to obtain form metadata, including a
    modified (preprocessed) form more easily manipulated by form
    compilers. The original form is left untouched. Currently, the
    following transformations are made to the preprocessed form:

      expand_compounds    (side effect of calling expand_derivatives)
      expand_derivatives
      renumber arguments and coefficients and apply evt. element mapping
    """
    times = []
    def tic(msg):
        times.append((time(), msg))

    tic('begin preprocess')

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting Form.")

    # Object names is empty if not given
    object_names = object_names or {}

    # Element mapping is empty if not given
    element_mapping = element_mapping or {}

    # Create empty form data
    form_data = FormData()

    # Store copies of preprocess input data
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
    # Temporary hack, don't touch:
    expand_derivatives = ufl.algorithms.ad.expand_derivatives
    form = expand_derivatives(form, common_cell.geometric_dimension())

    # Replace arguments and coefficients with new renumbered objects
    tic('extract_arguments_and_coefficients')
    original_arguments, original_coefficients = \
                        extract_arguments_and_coefficients(form)
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
    # Build mapping to original arguments and coefficients, which is
    # useful if the original arguments have data attached to them
    inv_replace_map = dict((w,v) for (v,w) in replace_map.iteritems())
    original_arguments = [inv_replace_map[v] for v in renumbered_arguments]
    original_coefficients = [inv_replace_map[w] for w in renumbered_coefficients]

    # Store data extracted by preprocessing
    if 1:
        # Not sure what to do with these later?
        form_data.arguments             = renumbered_arguments
        form_data.coefficients          = renumbered_coefficients
    form_data.original_arguments      = original_arguments
    form_data.original_coefficients   = original_coefficients
    form_data.renumbered_arguments    = renumbered_arguments
    form_data.renumbered_coefficients = renumbered_coefficients

    tic('replace')
    # FIXME: Always store mapping on the side instead of reconstructing
    if replace_functions:
        form = replace(form, replace_map)
        # Temporary hacks to introduce mappings in form compilers gradually
        form_data.element_replace_map = dict((e,e) for e in element_mapping.values())
        form_data.function_replace_map = dict((e,e) for e in replace_map.values())
    else:
        # Mappings from elements and functions (coefficients and arguments)
        # that reside in form to objects with canonical numbering as well as
        # completed cells and elements
        form_data.element_replace_map = element_mapping
        form_data.function_replace_map = replace_map

    # Store signature of form
    tic('signature')
    if skip_signature:
        form_data.signature = None
    else:
        form_data.signature = form.signature(form_data.function_replace_map)

    # Store elements, sub elements and element map
    tic('extract_elements')
    form_data.elements            = tuple(f.element() for f in
                                          chain(renumbered_arguments,
                                                renumbered_coefficients))
    form_data.unique_elements     = unique_tuple(form_data.elements)
    form_data.sub_elements        = extract_sub_elements(form_data.elements)
    form_data.unique_sub_elements = unique_tuple(form_data.sub_elements)

    # Store common cell
    form_data.cell = common_cell

    # Store data related to cell
    form_data.geometric_dimension = form_data.cell.geometric_dimension()
    form_data.topological_dimension = form_data.cell.topological_dimension()
    form_data.num_facets = form_data.cell.num_facets()

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

    # Store number of domains for integral types
    form_data.num_sub_domains = extract_num_sub_domains(form)
    (form_data.num_cell_domains,
     form_data.num_exterior_facet_domains,
     form_data.num_interior_facet_domains,
     form_data.num_point_domains,
     form_data.num_macro_cell_domains,
     form_data.num_surface_domains) = slice_dict(form_data.num_sub_domains,
                                                 _domain_types, 0)

    # Store number of domains for integral types
    form_data.domain_data = extract_domain_data(form)
    (form_data.cell_domain_data,
     form_data.exterior_facet_domain_data,
     form_data.interior_facet_domain_data,
     form_data.point_domain_data,
     form_data.macro_cell_domain_data,
     form_data.surface_domain_data) = slice_dict(form_data.domain_data,
                                                 _domain_types, None)

    # Store integrals stored by type and sub domain
    form_data.integral_data = extract_integral_data(form)

    # Store preprocessed form
    form._is_preprocessed = True
    form_data.preprocessed_form = form

    tic('end preprocess')

    # A coarse profiling implementation
    # TODO: Add counting of nodes
    # TODO: Add memory usage
    if preprocess.enable_profiling:
        print "-"*10, "UFL preprocess profiling:"
        for i in range(len(times)-1):
            t = times[i+1][0] - times[i][0]
            msg = times[i][1]
            print "%9.2e s    %s" % (t, msg)
        print "-"*60

    return form_data
preprocess.enable_profiling = False

class ExprData(object): # FIXME: Add __str__ operator etc like FormData
    pass

# FIXME: Rework preprocess_expression, following improvements to form preprocess
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

    use_expand_indices = True # TODO: make argument or fixate?

    # Check that we get an expression
    ufl_assert(isinstance(expr, Expr), "Expecting Expr.")

    # Create empty expression data
    expr_data = ExprData()

    # Store original expression
    expr_data.original_expr = expr

    # Get name of expr
    object_names = object_names or {}
    if id(expr) in object_names:
        name = object_names[id(expr)]
    else:
        name = "expr"

    # Element mapping is empty if not given
    element_mapping = element_mapping or {}

    # Extract common cell
    try:
        common_cell = extract_common_cell(expr, common_cell)
        gdim = common_cell.geometric_dimension()
    except:
        common_cell = Cell(None, None)
        gdim = None

    # Expand derivatives
    # Temporary hack, don't touch:
    expand_derivatives = ufl.algorithms.ad.expand_derivatives
    expr = expand_derivatives(expr, gdim)

    # Renumber indices
    if not use_expand_indices:
        expr = renumber_indices(expr)

    # Replace arguments and coefficients with new renumbered objects
    arguments, coefficients = extract_arguments_and_coefficients(expr)
    element_mapping = build_element_mapping(element_mapping, common_cell,
                                            arguments, coefficients)
    replace_map, arguments, coefficients = \
        build_argument_replace_map(arguments, coefficients, element_mapping)
    expr = replace(expr, replace_map)

    # Expand indices to simplify interpretation
    if use_expand_indices:
        expr = expand_indices(expr)

    # Build mapping to original arguments and coefficients, which is
    # useful if the original arguments have data attached to them
    inv_replace_map = dict((w,v) for (v,w) in replace_map.iteritems())
    original_arguments = [inv_replace_map[v] for v in arguments]
    original_coefficients = [inv_replace_map[w] for w in coefficients]

    # Store name of expr
    expr_data.name = name

    # Store data extracted by preprocessing
    expr_data.arguments             = arguments
    expr_data.coefficients          = coefficients
    expr_data.original_arguments    = original_arguments
    expr_data.original_coefficients = original_coefficients

    # Store signature of expression
    expr_data.signature = repr(expr)

    # Store elements, sub elements and element map
    expr_data.elements            = extract_elements(expr)
    expr_data.unique_elements     = unique_tuple(expr_data.elements)
    expr_data.sub_elements        = extract_sub_elements(expr_data.elements)
    expr_data.unique_sub_elements = unique_tuple(expr_data.sub_elements)

    # Store common cell
    expr_data.cell = common_cell

    # Store data related to cell
    if common_cell.is_undefined():
        expr_data.geometric_dimension = None
        expr_data.topological_dimension = None
    else:
        expr_data.geometric_dimension = expr_data.cell.geometric_dimension()
        expr_data.topological_dimension = expr_data.cell.topological_dimension()

    # Store some useful dimensions
    #expr_data.rank = len(expr_data.arguments)
    expr_data.num_coefficients = len(expr_data.coefficients)

    # Store argument names
    #expr_data.argument_names = \
    #    [object_names.get(id(expr_data.original_arguments[i]), "v%d" % i)
    #     for i in range(expr_data.rank)]

    # Store coefficient names
    expr_data.coefficient_names = \
        [object_names.get(id(expr_data.original_coefficients[i]), "w%d" % i)
         for i in range(expr_data.num_coefficients)]

    # Store preprocessed expression
    expr_data.preprocessed_expr = expr

    return expr_data


def extract_common_cell(form, common_cell=None):
    "Extract common cell for form or expression."

    # Either use given argument or try to find in form or expression
    common_cell = common_cell or form.cell()

    # Check common cell
    if common_cell is None or common_cell.is_undefined():
        error("Unable to extract common cell; "\
              "missing cell definition in form or expression.")

    return common_cell

def build_element_mapping(element_mapping, common_cell, arguments, coefficients):
    """Complete an element mapping for all elements used by
    arguments and coefficients, using a well defined common cell."""

    # Make a copy to avoid modifying the dict passed from non-ufl code
    element_mapping = dict(element_mapping)

    # Check that the given initial mapping has no invalid entries
    for v in element_mapping.itervalues():
        ufl_assert(not v.cell().is_undefined(),
                   "Found element with undefined cell in element mapping.")

    # Reconstruct all elements we need to map
    for f in chain(arguments, coefficients):
        e = f.element()
        if e in element_mapping:
            ufl_assert(not element_mapping[e].cell().is_undefined(),
                "Found element with undefined cell in given element mapping.")
        elif e.cell().is_undefined():
            ufl_assert(not common_cell.is_undefined(),
                "Cannot reconstruct elements with another undefined cell!")
            element_mapping[e] = e.reconstruct(cell=common_cell)

    return element_mapping
