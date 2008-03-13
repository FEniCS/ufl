#!/usr/bin/env python

"""
Functions to check properties of forms and integrals.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 13th 2008"

from ufl_io import *
from integral import *
from form import *


def is_multilinear(a):
    """Checks if a form is multilinear. Returns True/False."""
    ufl_assert(isinstance(o, Form), "Assuming a Form.")
    ufl_warning("is_multilinear is not implemented.")
    return True

