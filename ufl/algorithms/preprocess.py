"""This module provides the preprocess function which form compilers
will typically call prior to code generation to preprocess/simplify a
raw input form given by a user."""

__authors__ = "Anders Logg"
__date__ = "2009-12-07"

# Last changed: 2010-02-14

from ufl.log import info, warning, error
from ufl.assertions import ufl_assert
from ufl.form import Form

from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.transformations import replace
from ufl.algorithms.analysis import extract_arguments_and_coefficients, build_argument_replace_map
from ufl.algorithms.analysis import extract_elements, extract_sub_elements
from ufl.algorithms.analysis import extract_num_sub_domains, extract_integral_data, unique_tuple
from ufl.algorithms.formdata import FormData

def preprocess(form, object_names={}, common_cell=None):
    """
    Preprocess raw input form to obtain a form more easily manipulated
    by form compilers. The modified form is returned and the original
    form is left untouched. Currently, the following transformations
    are made to the modified form:

      expand_compounds    (side effect of calling expand_derivatives)
      expand_derivatives
      renumber_indices

    Form metadata is attached to the returned preprocessed form and may
    be accessed by calling form.form_data().
    """

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting Form.")

    # Check that form is not already preprocessed
    if form.form_data() is not None:
        info("Form is alreay preprocessed. Not updating form data.")
        return form

    # Get name of form
    if id(form) in object_names:
        name = object_names[id(form)]
    else:
        name = "a"

    # Expand derivatives
    form = expand_derivatives(form)

    # Renumber indices
    form = renumber_indices(form)

    # Replace arguments and coefficients with new renumbered objects
    arguments, coefficients = extract_arguments_and_coefficients(form)
    replace_map, arguments, coefficients = build_argument_replace_map(arguments, coefficients)
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

    # Store some useful dimensions
    form_data.rank = len(form_data.arguments)
    form_data.num_coefficients = len(form_data.coefficients)

    # Store argument names
    form_data.argument_names = [object_names.get(id(form_data.original_arguments[i]), "v%d" % i)
                                for i in range(form_data.rank)]

    # Store coefficient names
    form_data.coefficient_names = [object_names.get(id(form_data.original_coefficients[i]), "w%d" % i)
                                   for i in range(form_data.num_coefficients)]

    # Store elements, sub elements and element map
    form_data.elements            = extract_elements(form)
    form_data.unique_elements     = unique_tuple(form_data.elements)
    form_data.sub_elements        = extract_sub_elements(form_data.elements)
    form_data.unique_sub_elements = unique_tuple(form_data.sub_elements)

    # FIXME: Need to look at logic here, FFC does not support the last two cases

    # Store cell
    if not common_cell is None:
        form_data.cell = common_cell
    elif form_data.elements:
        cells = [element.cell() for element in form_data.elements]
        cells = [cell for cell in cells if not cell.domain() is None]
        if len(cells) == 0:
            error("Unable to extract form data. Reason: Missing cell definition in form.")
        form_data.cell = cells[0]
    elif form._integrals:
        # Special case to allow functionals only depending on geometric variables, with no elements
        form_data.cell = form._integrals[0].integrand().cell()
    else:
        # Special case to allow integral of constants to pass through without crashing
        form_data.cell = None
        warning("Form is empty, no elements or integrals, cell is undefined.")

    # Store data related to cell
    if form_data.cell is None:
        warning("No cell is defined in form.")
        form_data.geometric_dimension = None
        form_data.topological_dimension = None
        form_data.num_facets = None
    else:
        form_data.geometric_dimension = form_data.cell.geometric_dimension()
        form_data.topological_dimension = form_data.cell.topological_dimension()
        form_data.num_facets = form_data.cell.num_facets()

    # Store number of domains for integral types
    (form_data.num_cell_domains,
     form_data.num_exterior_facet_domains,
     form_data.num_interior_facet_domains,
     form_data.num_macro_cell_domains,
     form_data.num_surface_domains) = extract_num_sub_domains(form)

    # Store integrals stored by type and sub domain
    form_data.integral_data = extract_integral_data(form)

    # Attach form data to form
    form._form_data = form_data

    return form
