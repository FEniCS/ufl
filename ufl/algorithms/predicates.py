"""Functions to check properties of forms and integrals."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-09-29"

from ..output import ufl_error, ufl_warning, ufl_assert, ufl_info
from ..common import lstr
from ..form import Form
from ..base import Terminal
from ..basisfunction import BasisFunction
from ..algebra import Sum, Product
from .traversal import iter_expressions, traversal, post_traversal

#--- Utilities for checking properties of forms ---

def is_multilinear(form):
    "Check if form is multilinear."

    # FIXME: We don't check that each basis function appears to
    # FIXME: the same power in each term. For example, a = (v*u + v)*dx
    # FIXME: is not multilinear. Don't know how to check this without
    # FIXME: actually computing the monomial representation.

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Not a form: %s" % str(form))

    # Check that all operators are linear
    for e in iter_expressions(form):
        for (o, stack) in traversal(e):
            if not o.is_linear():
                return False

    return True
