"""This module provides the preprocess function which form compilers
will typically call prior to code generation to preprocess/simplify a
raw input form given by a user."""

# Copyright (C) 2009-2011 Anders Logg
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-12-07
# Last changed: 2011-05-02

from ufl.log import info, debug, warning, error
from ufl.assertions import ufl_assert
from ufl.form import Form

from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.transformations import replace
from ufl.algorithms.analysis import extract_arguments_and_coefficients, build_argument_replace_map
from ufl.algorithms.analysis import extract_elements, extract_sub_elements
from ufl.algorithms.analysis import extract_num_sub_domains, extract_integral_data, unique_tuple
from ufl.algorithms.formdata import FormData

def preprocess(form, object_names=None, common_cell=None):
    """
    Preprocess raw input form to obtain form metadata, including a
    modified (preprocessed) form more easily manipulated by form
    compilers. The original form is left untouched. Currently, the
    following transformations are made to the preprocessed form:

      expand_compounds    (side effect of calling expand_derivatives)
      expand_derivatives
      renumber_indices
    """

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting Form.")

    # Get name of form
    object_names = object_names or {}
    if id(form) in object_names:
        name = object_names[id(form)]
    else:
        name = "a"

    # Extract common cell
    common_cell = common_cell or form.cell()

    # Check common cell
    if common_cell is None or common_cell.is_undefined():
        error("""\
Unable to extract common cell; missing cell definition in form.""")

    # Expand derivatives
    form = expand_derivatives(form, common_cell.geometric_dimension())

    # Renumber indices
    form = renumber_indices(form)

    # Replace arguments and coefficients with new renumbered objects
    arguments, coefficients = extract_arguments_and_coefficients(form)
    replace_map, arguments, coefficients = \
        build_argument_replace_map(arguments, coefficients)
    form = replace(form, replace_map)

    # Build mapping to original arguments and coefficients, which is
    # useful if the original arguments have data attached to them
    inv_replace_map = {}
    for v, w in replace_map.iteritems():
        inv_replace_map[w] = v
    original_arguments = [inv_replace_map[v] for v in arguments]
    original_coefficients = [inv_replace_map[v] for v in coefficients]

    # Create empty form data
    form_data = FormData()

    # Store name of form
    form_data.name = name

    # Store data extracted by preprocessing
    form_data.arguments             = arguments
    form_data.coefficients          = coefficients
    form_data.original_arguments    = original_arguments
    form_data.original_coefficients = original_coefficients

    # Store signature of form
    form_data.signature = form.signature()

    # Store elements, sub elements and element map
    form_data.elements            = extract_elements(form)
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
    form_data.rank = len(form_data.arguments)
    form_data.num_coefficients = len(form_data.coefficients)

    # Store argument names
    form_data.argument_names = \
        [object_names.get(id(form_data.original_arguments[i]), "v%d" % i)
         for i in range(form_data.rank)]

    # Store coefficient names
    form_data.coefficient_names = \
        [object_names.get(id(form_data.original_coefficients[i]), "w%d" % i)
         for i in range(form_data.num_coefficients)]

    # Store number of domains for integral types
    (form_data.num_cell_domains,
     form_data.num_exterior_facet_domains,
     form_data.num_interior_facet_domains,
     form_data.num_macro_cell_domains,
     form_data.num_surface_domains) = extract_num_sub_domains(form)

    # Store integrals stored by type and sub domain
    form_data.integral_data = extract_integral_data(form)

    # Store preprocessed form
    form._is_preprocessed = True
    form_data.preprocessed_form = form

    return form_data
