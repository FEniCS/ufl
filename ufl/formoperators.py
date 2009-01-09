"Various high level ways to transform a complete Form into a new Form."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-01-09"

from ufl.form import Form
from ufl.terminal import Tuple
from ufl.function import Function
from ufl.basisfunction import BasisFunction, BasisFunctions
from ufl.differentiation import FunctionDerivative

# An exception to the rule that ufl.* does not depend on ufl.algorithms.* ...
from ufl.algorithms import compute_form_adjoint, compute_form_action, \
                           compute_form_lhs, compute_form_rhs

def rhs(form):
    """Given a combined bilinear and linear form,
    extract the linear form part (right hand side).

    TODO: Given "a = u*v*dx + f*v*dx, should this
    return "+f*v*dx" as found in the form or
    "-f*v*dx" as the rigth hand side should
    be when solving the equations?
    """
    # TODO: May need to compute form derivatives before applying this!
    return compute_form_rhs(form)

def lhs(form):
    """Given a combined bilinear and linear form,
    extract the bilinear form part (left hand side)."""
    # TODO: May need to compute form derivatives before applying this!
    return compute_form_lhs(form)

def action(form, function=None):
    """Given a bilinear form, return a linear form
    with an additional function coefficient, representing
    the action of the form on the function. This can be
    used for matrix-free methods."""
    # TODO: May need to compute form derivatives before applying this!
    return compute_form_action(form, function)

def adjoint(form):
    """Given a combined bilinear form, compute the adjoint
    form by swapping the test and trial functions."""
    # TODO: May need to compute form derivatives before applying this!
    return compute_form_adjoint(form)

def _handle_derivative_arguments(function, basisfunction):
    if isinstance(function, Function):
        functions = (function,)
        
        # Get element
        element = function.element()
        
        # Create basis function if necessary
        if basisfunction is None:
            basisfunctions = BasisFunction(element)
        else:
            basisfunctions = (basisfunction,)
    
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
        if basisfunction is None:
            basisfunctions = BasisFunctions(element)
        else:
            basisfunctions = (basisfunction,)
            ufl_assert(isinstance(basisfunction, BasisFunction) \
                and basisfunction.element() == element,
                "Basis function over wrong element supplied, "\
                "got %s but expecting %s." % \
                (repr(basisfunction.element()), repr(element)))
    
    functions      = Tuple(functions)
    basisfunctions = Tuple(basisfunctions)
    
    return functions, basisfunctions

def derivative(form, function, basisfunction=None):
    """Given any form, compute the linearization of the
    form with respect to the given discrete function.
    The resulting form has one additional basis function
    in the same finite element space as the function.
    A tuple of Functions may be provided in place of
    a single Function, in which case the new BasisFunction
    argument is based on a MixedElement created from this tuple."""
    
    functions, basisfunctions = _handle_derivative_arguments(function, basisfunction)
    
    # Got a form? Apply derivatives to the integrands in turn.
    if isinstance(form, Form):
        integrals = []
        for itg in form._integrals:
            fd = FunctionDerivative(itg.integrand(), functions, basisfunctions)
            newitg = itg.reconstruct(integrand=fd)
            integrals.append(newitg)
        return Form(integrals)
    
    elif isinstance(form, Expr):
        # What we got was in fact an integrand
        return FunctionDerivative(form, functions, basisfunctions)
    
    ufl_error("Invalid argument type %s." % str(type(form)))

