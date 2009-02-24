"Various high level ways to transform a complete Form into a new Form."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-01-16"

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.expr import Expr
from ufl.terminal import Tuple
from ufl.finiteelement import MixedElement
from ufl.function import Function
from ufl.basisfunction import BasisFunction, BasisFunctions
from ufl.differentiation import FunctionDerivative

# An exception to the rule that ufl.* does not depend on ufl.algorithms.* ...
from ufl.algorithms import compute_form_adjoint, compute_form_action, \
                           compute_form_lhs, compute_form_rhs, compute_form_functional, expand_derivatives

def lhs(form):
    """Given a combined bilinear and linear form,
    extract the bilinear form part (left hand side)."""
    form = expand_derivatives(form)
    return compute_form_lhs(form)

def rhs(form):
    """Given a combined bilinear and linear form,
    extract the linear form part (right hand side).

    TODO: Given "a = u*v*dx + f*v*dx, should this
    return "+f*v*dx" as found in the form or
    "-f*v*dx" as the rigth hand side should
    be when solving the equations?
    """
    form = expand_derivatives(form)
    return compute_form_rhs(form)

def functional(form): # TODO: Does this make sense for anything other than testing?
    """Extract the functional part of form."""
    form = expand_derivatives(form)
    return compute_form_functional(form)

def action(form, function=None):
    """Given a bilinear form, return a linear form
    with an additional function coefficient, representing
    the action of the form on the function. This can be
    used for matrix-free methods."""
    form = expand_derivatives(form)
    return compute_form_action(form, function)

def adjoint(form):
    """Given a combined bilinear form, compute the adjoint
    form by swapping the test and trial functions."""
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
            basis_functions = (basis_function,)
            ufl_assert(isinstance(basis_function, BasisFunction) \
                and basis_function.element() == element,
                "Basis function over wrong element supplied, "\
                "got %s but expecting %s." % \
                (repr(basis_function.element()), repr(element)))
    
    functions      = Tuple(*functions)
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

