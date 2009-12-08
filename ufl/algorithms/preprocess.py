"""This module provides the preprocess function which form compilers
will typically call prior to code generation to preprocess/simplify a
raw input form given by a user."""

__authors__ = "Anders Logg"
__date__ = "2009-12-07"

# Last changed: 2009-12-08

from ufl.log import info
from ufl.assertions import ufl_assert
from ufl.form import Form

from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.analysis import extract_arguments_and_coefficients, build_argument_replace_map
from ufl.algorithms.transformations import replace

def preprocess(form):
    """Preprocess raw input form to obtain a form more easily
    manipulated by form compilers. The modified form is returned and
    the original form is left untouched. Currently, the following
    transformations are made to the modified form:

      expand_compounds    (side effect of calling expand_derivatives)
      expand_derivatives
      renumber_indices
    """

    info("Preprocessing form")

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting Form.")

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

    # Store data for later extraction
    form._form_data = (arguments,
                       coefficients,
                       original_arguments,
                       original_coefficients)

    return form
