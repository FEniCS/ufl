"""Functions to check the validity of forms."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-03-14"

# Modified by Anders Logg, 2008-2009.
# Modified by Mehdi Nikbakht, 2010.
# Last changed: 2010-09-29

from ufl.log import warning
from ufl.assertions import ufl_assert

# UFL classes
from ufl.form import Form
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.constantvalue import is_true_ufl_scalar
from ufl.integral import Measure

# UFL algorithms
from ufl.algorithms.preprocess import preprocess
from ufl.algorithms.traversal import iter_expressions, traverse_terminals, fast_pre_traversal
from ufl.algorithms.analysis import extract_elements
from ufl.algorithms.predicates import is_multilinear
from ufl.algorithms.propagate_restrictions import check_restrictions

def validate_form(form): # TODO: Can we make this return a list of errors instead of raising exception?
    """Performs all implemented validations on a form. Raises exception if something fails."""

    ufl_assert(isinstance(form, Form), "Expecting a Form.")

    # Preprocess form
    #form = preprocess(form)

    # FIXME: Add back check for multilinearity
    # Check that form is multilinear
    #ufl_assert(is_multilinear(form), "Form is not multilinear in basis function arguments.")

    # Check that cell is the same everywhere
    cells = set()
    for e in iter_expressions(form):
        cells.update(t.cell() for t in traverse_terminals(e) if not (t.cell() is None or t.cell().domain() is None))
    if None in cells:
        cells.remove(None)
    ufl_assert(len(cells) <= 1,
               "Inconsistent cell definitions in form: %s." % str(cells))

    # Check that no Coefficient or Argument instance
    # have the same count unless they are the same
    functions = {}
    arguments = {}
    for e in iter_expressions(form):
        for f in traverse_terminals(e):
            if isinstance(f, Coefficient):
                c = f.count()
                if c in functions:
                    g = functions[c]
                    ufl_assert(f is g, "Got different Coefficients with same count: %s and %s." % (repr(f), repr(g)))
                else:
                    functions[c] = f

            elif isinstance(f, Argument):
                c = f.count()
                if c in arguments:
                    g = arguments[c]
                    if c == -2: msg = "TestFunctions"
                    elif c == -1: msg = "TrialFunctions"
                    else: msg = "Arguments with same count"
                    msg = "Got different %s: %s and %s." % (msg, repr(f), repr(g))
                    ufl_assert(f is g, msg)
                else:
                    arguments[c] = f

    # Check that all integrands are scalar
    for expression in iter_expressions(form):
        ufl_assert(is_true_ufl_scalar(expression),
            "Got non-scalar integrand expression:\n%s\n%s" % (str(expression), repr(expression)))

    # Check that restrictions are permissible
    for integral in form.integrals():
        # Only allow restricitions on interior facet integrals and surface measures
        if integral.measure().domain_type() == Measure.INTERIOR_FACET or integral.measure().domain_type() == Measure.SURFACE:
            check_restrictions(integral.integrand(), True)
        else:
            check_restrictions(integral.integrand(), False)
