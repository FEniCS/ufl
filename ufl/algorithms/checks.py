"""Functions to check the validity of forms."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-01-16"

# Modified by Anders Logg, 2008.

from ufl.output import ufl_assert, ufl_warning

# All classes:
from ufl.form import Form
from ufl.scalar import is_true_ufl_scalar

# Other algorithms
from ufl.algorithms.traversal import post_traversal, post_walk, iter_expressions, traverse_terminals
from ufl.algorithms.analysis import extract_elements
from ufl.algorithms.predicates import is_multilinear
from ufl.algorithms.ad import expand_derivatives

def validate_form(form):
    """Performs all implemented validations on a form. Raises exception if something fails."""
    
    ufl_assert(isinstance(form, Form), "Expecting a Form.")
    
    # TODO: Can we check for multilinearity without expanding function derivatives?
    form = expand_derivatives(form)
    ufl_assert(is_multilinear(form), "Form is not multilinear in basis function arguments.")
    #if not is_multilinear(form): ufl_warning("Form is not multilinear.")
    
    # Check that cell is the same everywhere
    cells = set()
    for e in iter_expressions(form):
        cells.update(t.cell() for t in traverse_terminals(e))
    if None in cells:
        cells.remove(None)
    ufl_assert(len(cells) == 1,
        "Inconsistent or missing cell definitions in form, found %s." % str(cells))
    
    # Check that all integrands are scalar
    for expression in iter_expressions(form):
        ufl_assert(is_true_ufl_scalar(expression),
            "Got non-scalar integrand expression:\n%s\n%s" % (str(expression), repr(expression)))

