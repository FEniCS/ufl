"""This module defines utilities for transforming
complete Forms into new related Forms."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-01 -- 2008-10-02"

from collections import defaultdict
from itertools import izip

from ..common import some_key, product
from ..output import ufl_assert, ufl_error, ufl_warning

# All classes:
from ..basisfunction import BasisFunction
#from ..basisfunction import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..function import Function, Constant
from ..form import Form
from ..integral import Integral

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients
from .transformations import replace, replace_in_form


def compute_form_action(form, function):
    """Compute the action of a form on a Function.
    
    This works simply by replacing the last basisfunction
    with a Function on the same function space (element).
    The form returned will thus have one BasisFunction less 
    and one additional Function at the end.
    """
    bf = basisfunctions(form)
    ufl_assert(len(bf) == 2, "Expecting bilinear form.")
    v, u = bf
    e = u.element()
    if function is None:
        function = Function(e)
    else:
        ufl_assert(function.element() == e, \
            "Trying to compute action of form on a "\
            "function in an incompatible element space.")
    return replace_in_form(form, {u:function})


def compute_form_lhs(form):
    """Compute the left hand side of a form."""
    #FIXME


def compute_form_rhs(form):
    """Compute the right hand side of a form."""
    #FIXME

def compute_form_transpose(form):
    """Compute the transpose of a form.
    
    This works simply by swapping the first and last basisfunctions.
    """
    bf = basisfunctions(form)
    ufl_assert(len(bf) == 2, "Expecting bilinear form.")
    v, u = bf
    return replace_in_form(form, {v:u, u:v})


def compute_dual_form(form): # FIXME: Don't know if this is correct, or if we need it?
    """Compute the dual of a bilinear form:
    a(v,u;...) -> a(u,v;...)
    
    a(v,u) = \int_\Omega u*v dx + \int_\Gamma f*v dx
    
    This assumes a bilinear form and works simply by
    replacing the trial function with the test function.
    The form returned will thus be a linear form.
    """
    bf = basisfunctions(form)
    ufl_assert(len(bf) == 2, "Expecting bilinear form.")
    v, u = bf
    return replace_in_form(form, {u:v, v:u})


def compute_dirichlet_functional(form): # FIXME: Don't know if this is correct or even useful, just picked up the name some place.
    """Compute the Dirichlet functional of a form:
    a(v,u;...) - L(v; ...) -> 0.5 a(v,v;...) - L(v;...)
    
    This assumes a bilinear form and works simply by
    replacing the trial function with the test function.
    The form returned will thus be a linear form.
    """
    return 0.5*compute_form_lhs(form) - compute_form_rhs(form)
    #bf = basisfunctions(form)
    #ufl_assert(len(bf) == 2, "Expecting bilinear form.")
    #v, u = bf
    #return replace_in_form(form, {u:v})

