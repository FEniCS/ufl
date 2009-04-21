"Various high level ways to transform a complete Form into a new Form."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-04-20"

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.expr import Expr
from ufl.split_functions import split
from ufl.terminal import Tuple
from ufl.variable import Variable
from ufl.finiteelement import MixedElement
from ufl.function import Function
from ufl.basisfunction import BasisFunction, BasisFunctions
from ufl.differentiation import FunctionDerivative
from ufl.constantvalue import is_true_ufl_scalar

# An exception to the rule that ufl.* does not depend on ufl.algorithms.* ...
from ufl.algorithms import compute_form_adjoint, \
                           compute_form_action, \
                           compute_energy_norm, \
                           compute_form_lhs, \
                           compute_form_rhs, \
                           compute_form_functional, \
                           expand_derivatives, \
                           as_form

from ufl.algorithms import replace

def lhs(form):
    """Given a combined bilinear and linear form,
    extract the left hand side (bilinear form part).

    Example:

        a = u*v*dx + f*v*dx
        a = lhs(a) -> u*v*dx
    """
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_lhs(form)

def rhs(form):
    """Given a combined bilinear and linear form,
    extract the right hand side (negated linear form part)."

    Example:

        a = u*v*dx + f*v*dx
        L = rhs(a) -> -f*v*dx
    """
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_rhs(form)

def system(form):
    "Split a form into the left hand side and right hand side, see lhs and rhs."
    return lhs(form), rhs(form)

def functional(form): # TODO: Does this make sense for anything other than testing?
    """Extract the functional part of form."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_functional(form)

def action(form, function=None):
    """Given a bilinear form, return a linear form
    with an additional function coefficient, representing
    the action of the form on the function. This can be
    used for matrix-free methods."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_action(form, function)

def energy_norm(form, function=None):
    """Given a bilinear form, return a linear form
    with an additional function coefficient, representing
    the action of the form on the function. This can be
    used for matrix-free methods."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_energy_norm(form, function)

def adjoint(form):
    """Given a combined bilinear form, compute the adjoint
    form by swapping the test and trial functions."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_adjoint(form)

def _handle_derivative_arguments(function, basis_function):
    if isinstance(function, Function):
        functions = (function,)
        
        # Get element
        element = function.element()
        
        # Create basis function if necessary
        if basis_function is None:
            basis_functions = (BasisFunction(element),)
        else:
            basis_functions = (basis_function,)
    
    elif isinstance(function, tuple):
        functions = function
        
        # We got a tuple of functions, handle it as 
        # functions over components of a mixed element.
        ufl_assert(all(isinstance(w, Function) for w in functions),
            "Expecting a tuple of Functions to differentiate w.r.t.")
        
        # Create mixed element
        elements = [w.element() for w in functions]
        element = MixedElement(*elements)
        
        # Create basis functions if necessary
        if basis_function is None:
            basis_functions = BasisFunctions(element)
        else:
            basis_functions = split(basis_function)
            ufl_assert(isinstance(basis_function, BasisFunction) \
                and basis_function.element() == element,
                "Basis function over wrong element supplied, "\
                "got %s but expecting %s." % \
                (repr(basis_function.element()), repr(element)))
    
    functions       = Tuple(*functions)
    basis_functions = Tuple(*basis_functions)
    
    return functions, basis_functions

def derivative(form, function, basis_function=None):
    """Given any form, compute the linearization of the
    form with respect to the given discrete function.
    The resulting form has one additional basis function
    in the same finite element space as the function.
    A tuple of Functions may be provided in place of
    a single Function, in which case the new BasisFunction
    argument is based on a MixedElement created from this tuple."""
    
    functions, basis_functions = _handle_derivative_arguments(function, basis_function)
    
    # Got a form? Apply derivatives to the integrands in turn.
    if isinstance(form, Form):
        integrals = []
        for itg in form._integrals:
            fd = FunctionDerivative(itg.integrand(), functions, basis_functions)
            integrals.append(itg.reconstruct(fd))
        return Form(integrals)
    
    elif isinstance(form, Expr):
        # What we got was in fact an integrand
        return FunctionDerivative(form, functions, basis_functions)
    
    error("Invalid argument type %s." % str(type(form)))

def sensitivity_rhs(a, u, L, v):
    """Compute the right hand side for a sensitivity calculation system.

    The derivation behind this computation is as follows.
    Assume a, L to be bilinear and linear forms
    corresponding to the assembled linear system

        Ax = b.

    Where x is the vector of the discrete function corresponding to u.
    Let v be some scalar variable this equation depends on.
    Then we can write
        
        0 = d/dv[Ax-b] = dA/dv x + A dx/dv - db/dv,
        A dx/dv = db/dv - dA/dv x,
    
    and solve this system for dx/dv, using the same bilinear form a
    and matrix A from the original system.
    Assume the forms are written
    
        v = variable(v_expression)
        L = IL(v)*dx
        a = Ia(v)*dx
    
    where IL and Ia are integrand expressions.
    Define a Function u representing the solution
    to the equations. Then we can compute db/dv
    and dA/dv from the forms

        da = diff(a, v)
        dL = diff(L, v)

    and the action of da on u by

        dau = action(da, u)

    In total, we can build the right hand side of the system
    to compute du/dv with the single line
        
        dL = diff(L, v) - action(diff(a, v), u)

    or, using this function

        dL = sensitivity_rhs(a, u, L, v)
    """
    msg = "Expecting (a, u, L, v), (bilinear form, function, linear form and scalar variable)."
    ufl_assert(isinstance(a, Form), msg)
    ufl_assert(isinstance(u, Function), msg)
    ufl_assert(isinstance(L, Form), msg)
    ufl_assert(isinstance(v, Variable), msg)
    ufl_assert(is_true_ufl_scalar(v), "Expecting scalar variable.")
    from ufl.operators import diff
    return diff(L, v) - action(diff(a, v), u)

