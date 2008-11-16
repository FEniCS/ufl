"""This module defines utilities for transforming
complete Forms into new related Forms."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-01 -- 2008-10-30"

# Modified by Anders Logg, 2008

from itertools import izip

from ufl.common import some_key, product
from ufl.output import ufl_assert, ufl_error, ufl_warning

# All classes:
from ufl.basisfunction import BasisFunction
#from ufl.basisfunction import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ufl.function import Function, Constant
from ufl.form import Form
from ufl.integral import Integral

# Lists of all Expr classes
from ufl.classes import ufl_classes, terminal_classes, nonterminal_classes

# Other algorithms:
from ufl.algorithms.analysis import extract_basisfunctions, extract_coefficients
from ufl.algorithms.transformations import replace, replace_in_form


def compute_form_action(form, function):
    """Compute the action of a form on a Function.
    
    This works simply by replacing the last basisfunction
    with a Function on the same function space (element).
    The form returned will thus have one BasisFunction less 
    and one additional Function at the end.
    """
    bf = extract_basisfunctions(form)
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
    # TODO: Can we use extract_basisfunction_dependencies for this?
    ufl_error("Not implemented.")

def compute_form_rhs(form):
    """Compute the right hand side of a form."""
    # TODO: Can we use extract_basisfunction_dependencies for this?
    ufl_error("Not implemented.")

def compute_form_adjoint(form):
    """Compute the adjoint of a bilinear form.
    
    This works simply by swapping the first and last basisfunctions.
    """
    bf = extract_basisfunctions(form)
    ufl_assert(len(bf) == 2, "Expecting bilinear form.")
    v, u = bf
    return replace_in_form(form, {v:u, u:v})

#def compute_dirichlet_functional(form):
#    """Compute the Dirichlet functional of a form:
#    a(v,u;...) - L(v; ...) -> 0.5 a(v,v;...) - L(v;...)
#    
#    This assumes a bilinear form and works simply by
#    replacing the trial function with the test function.
#    The form returned will thus be a linear form.
#    """
#    ufl_warning("TODO: Don't know if this is correct or even useful, just picked up the name some place.")
#    return 0.5*compute_form_lhs(form) - compute_form_rhs(form)
#    #bf = extract_basisfunctions(form)
#    #ufl_assert(len(bf) == 2, "Expecting bilinear form.")
#    #v, u = bf
#    #return replace_in_form(form, {u:v})
