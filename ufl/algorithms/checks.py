"""Functions to check the validity of forms."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-23"

# Modified by Anders Logg, 2008.

from ..output import UFLException, ufl_error, ufl_assert, ufl_info, ufl_warning
from ..base import is_python_scalar, is_ufl_scalar, is_true_ufl_scalar

# All classes:
from ..form import Form
from ..finiteelement import _domain2dim

# Other algorithms
from .traversal import post_traversal, post_walk, iter_expressions
from .analysis import extract_value_shape, extract_domain, extract_elements
from .predicates import is_multilinear

def validate_form(form):
    """Performs all implemented validations on a form. Raises exception if something fails."""

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting a Form.")

    # Check that form is multilinear
    is_ml = is_multilinear(form)
    #if not is_ml: ufl_warning("Form is not multilinear.")
    ufl_assert(is_ml, "Form is not multilinear.")

    # Check that domain is the same for all elements
    domain = extract_domain(form)
    for element in extract_elements(form):
        ufl_assert(domain == element.domain(), "Inconsistent domains in form, got both %s and %s." % (domain, element.domain()))

    # Check that all integrands are scalar
    dim = _domain2dim[domain]
    for expression in iter_expressions(form):
        ufl_assert(extract_value_shape(expression, dim) == (), "Got non-scalar integrand expression:\n%s" % expression)
