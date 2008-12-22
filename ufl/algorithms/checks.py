"""Functions to check the validity of forms."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-12-22"

# Modified by Anders Logg, 2008.

from ufl.output import ufl_assert, ufl_warning

# All classes:
from ufl.form import Form

# Other algorithms
from ufl.algorithms.traversal import post_traversal, post_walk, iter_expressions, traverse_terminals
from ufl.algorithms.analysis import extract_elements
from ufl.algorithms.predicates import is_multilinear

def validate_form(form):
    """Performs all implemented validations on a form. Raises exception if something fails."""

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting a Form.")

    # Check that form is multilinear
    is_ml = is_multilinear(form)
    #if not is_ml: ufl_warning("Form is not multilinear.")
    ufl_assert(is_ml, "Form is not multilinear.")
    
    # Check that cell is the same everywhere
    cells = set()
    for e in iter_expressions(form):
        for t in traverse_terminals(e):
            cells.add(t.cell())
    if None in cells:
        cells.remove(None)
    ufl_assert(len(cells) == 1, "Inconsistent or missing cell in form, found %s." % str(cells))
    
    # Check that all integrands are scalar
    for expression in iter_expressions(form):
        ufl_assert(expression.shape() == (), "Got non-scalar integrand expression:\n%s" % expression)
