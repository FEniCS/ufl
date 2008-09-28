"""Functions to check properties of forms and integrals."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-09-28"

from ..output import ufl_error, ufl_warning, ufl_assert, ufl_info
from ..form import Form
from ..basisfunction import BasisFunction
from .traversal import iter_expressions, pre_traversal

#--- Utilities for checking properties of forms ---

def is_multilinear(form):
    "Check if form is multilinear"

    ufl_assert(isinstance(form, Form), "Not a form: %s" % str(form))
    ufl_warning("is_multilinear does not work yet (in preparation)")

    # Iterate over integrands
    for e in iter_expressions(form):

        # Iterate over operands
        for o in pre_traversal(e):
            if not o.is_linear():
                return False

    # FIXME: Check arity of each term, must be equal

    return True
