"Various high level ways to transform a complete Form into a new Form."

# Copyright (C) 2008-2011 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009
#
# First added:  2008-03-14
# Last changed: 2010-09-07

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.expr import Expr
from ufl.split_functions import split
from ufl.terminal import Tuple
from ufl.variable import Variable
from ufl.finiteelement import MixedElement
from ufl.argument import Argument, Arguments
from ufl.coefficient import Coefficient
from ufl.differentiation import CoefficientDerivative
from ufl.constantvalue import is_true_ufl_scalar

# An exception to the rule that ufl.* does not depend on ufl.algorithms.* ...
from ufl.algorithms import compute_form_adjoint, \
                           compute_form_action, \
                           compute_energy_norm, \
                           compute_form_lhs, \
                           compute_form_rhs, \
                           compute_form_functional, \
                           expand_derivatives, \
                           expand_indices, \
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
    #form = expand_indices(form)
    return compute_form_lhs(form)

def rhs(form):
    """Given a combined bilinear and linear form,
    extract the right hand side (negated linear form part).

    Example:

        a = u*v*dx + f*v*dx
        L = rhs(a) -> -f*v*dx
    """

    form = as_form(form)
    form = expand_derivatives(form)
    #form = expand_indices(form)
    return compute_form_rhs(form)

def system(form):
    "Split a form into the left hand side and right hand side, see lhs and rhs."
    return lhs(form), rhs(form)

def functional(form): # TODO: Does this make sense for anything other than testing?
    """Extract the functional part of form."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_functional(form)

def action(form, coefficient=None):
    """Given a bilinear form, return a linear form
    with an additional coefficient, representing the
    action of the form on the coefficient. This can be
    used for matrix-free methods."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_action(form, coefficient)

def energy_norm(form, coefficient=None):
    """Given a bilinear form, return a linear form
    with an additional coefficient, representing the
    action of the form on the coefficient. This can be
    used for matrix-free methods."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_energy_norm(form, coefficient)

def adjoint(form):
    """Given a combined bilinear form, compute the adjoint
    form by swapping the test and trial functions."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_adjoint(form)

def _handle_derivative_arguments(coefficient, argument):
    """Valid combinations:
    - Coefficient, Argument. Elements must match.
    - (Coefficient tuple,), Argument. Argument element must be a mixed element with subelements matching elements of the Coefficient tuple.
    """

    if isinstance(coefficient, Coefficient):
        # Place in generic tuple
        coefficients = (coefficient,)

        # Get element
        element = coefficient.element()

        # Create Argument if necessary
        if argument is None:
            argument = Argument(element)
        ufl_assert(isinstance(argument, Argument),
            "Expecting Argument instance, not %s." % type(argument))
        ufl_assert(argument.element() == element,
            "Argument over wrong element supplied, "\
            "got %s but expecting %s." % \
            (repr(argument.element()), repr(element)))

        # Place in generic tuple
        arguments = (argument,)

    elif isinstance(coefficient, (tuple, list)):
        # Place in generic tuple
        coefficients = coefficient

        # We got a tuple of coefficients, handle it as
        # coefficients over components of a mixed element.
        ufl_assert(all(isinstance(w, Coefficient) for w in coefficients),
            "Expecting a tuple of Coefficients to differentiate w.r.t.")

        # Create mixed element
        elements = [w.element() for w in coefficients]
        element = MixedElement(*elements)

        # Create arguments if necessary
        if argument is None:
            argument = Argument(element)
        ufl_assert(isinstance(argument, Argument),
            "Expecting Argument instance, not %s." % type(argument))
        ufl_assert(argument.element() == element,
            "Arguments over wrong element supplied, "\
            "got %s but expecting %s." % \
            (repr(argument.element()), repr(element)))

        # Place in generic tuple
        arguments = split(argument)

    else:
        error("Expecting Coefficient instance or tuple of Coefficient instances, not %s." % type(coefficient))

    # Wrap and return generic tuples
    coefficients       = Tuple(*coefficients)
    arguments = Tuple(*arguments)
    return coefficients, arguments

def derivative(form, coefficient, argument=None):
    """Given any form, compute the linearization of the
    form with respect to the given Coefficient.
    The resulting form has one additional Argument
    in the same finite element space as the coefficient.
    A tuple of Coefficients may be provided in place of
    a single Coefficient, in which case the new Argument
    argument is based on a MixedElement created from this tuple."""

    coefficients, arguments = _handle_derivative_arguments(coefficient, argument)

    # Got a form? Apply derivatives to the integrands in turn.
    if isinstance(form, Form):
        integrals = []
        for itg in form._integrals:
            fd = CoefficientDerivative(itg.integrand(), coefficients, arguments)
            integrals.append(itg.reconstruct(fd))
        return Form(integrals)

    elif isinstance(form, Expr):
        # What we got was in fact an integrand
        return CoefficientDerivative(form, coefficients, arguments)

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
    Define a Coefficient u representing the solution
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
    ufl_assert(isinstance(u, Coefficient), msg)
    ufl_assert(isinstance(L, Form), msg)
    ufl_assert(isinstance(v, Variable), msg)
    ufl_assert(is_true_ufl_scalar(v), "Expecting scalar variable.")
    from ufl.operators import diff
    return diff(L, v) - action(diff(a, v), u)

