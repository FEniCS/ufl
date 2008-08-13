"""Functions to check properties of forms and integrals."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-13"

from ..output import ufl_error, ufl_warning, ufl_assert, ufl_info
from ..all import Form


#--- Utilities for checking properties of forms ---

def is_multilinear(a):
    """Checks if a form is multilinear. Returns True/False."""
    ufl_assert(isinstance(o, Form), "Assuming a Form.")
    ufl_warning("is_multilinear is not implemented.")
    return True

