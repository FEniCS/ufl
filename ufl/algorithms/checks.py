"""Functions to check the validity of forms."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-03"

# Modified by Anders Logg, 2008.

from ufl.log import warning
from ufl.assertions import ufl_assert

# UFL classes:
from ufl.form import Form
from ufl.function import Function
from ufl.basisfunction import BasisFunction
from ufl.constantvalue import is_true_ufl_scalar

# Other algorithms
from ufl.algorithms.traversal import iter_expressions, traverse_terminals
from ufl.algorithms.analysis import extract_elements
from ufl.algorithms.predicates import is_multilinear
from ufl.algorithms.ad import expand_derivatives


def validate_form(form):
    """Performs all implemented validations on a form. Raises exception if something fails."""
    
    ufl_assert(isinstance(form, Form), "Expecting a Form.")
    
    # TODO: Can we check for multilinearity without expanding function derivatives?
    form = expand_derivatives(form)
    ufl_assert(is_multilinear(form), "Form is not multilinear in basis function arguments.")
    #if not is_multilinear(form): warning("Form is not multilinear.")
    
    # Check that cell is the same everywhere
    cells = set()
    for e in iter_expressions(form):
        cells.update(t.cell() for t in traverse_terminals(e))
    if None in cells:
        cells.remove(None)
    ufl_assert(len(cells) == 1,
        "Inconsistent or missing cell definitions in form, found %s." % str(cells))
    
    # Check that no Function or BasisFunction instance
    # have the same count unless they are the same
    functions = {}
    basisfunctions = {}
    for e in iter_expressions(form):
        for f in traverse_terminals(e):
            if isinstance(f, Function):
                c = f.count()
                if c in functions:
                    g = functions[c]
                    ufl_assert(f is g, "Got different Functions with same count: %s and %s." % (repr(f), repr(g)))
                else:
                    functions[c] = f
            
            elif isinstance(f, BasisFunction):
                c = f.count()
                if c in basisfunctions:
                    g = basisfunctions[c]
                    if c == -2: msg = "TestFunctions"
                    elif c == -1: msg = "TrialFunctions"
                    else: msg = "BasisFunctions with same count"
                    msg = "Got different %s: %s and %s." % (msg, repr(f), repr(g))
                    ufl_assert(f is g, msg)
                else:
                    basisfunctions[c] = f
            
    # Check that all integrands are scalar
    for expression in iter_expressions(form):
        ufl_assert(is_true_ufl_scalar(expression),
            "Got non-scalar integrand expression:\n%s\n%s" % (str(expression), repr(expression)))
    
