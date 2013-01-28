"Various high level ways to transform a complete Form into a new Form."

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Last changed: 2013-01-02

from itertools import izip
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.form import Form
from ufl.expr import Expr
from ufl.split_functions import split
from ufl.operatorbase import Tuple
from ufl.variable import Variable
from ufl.finiteelement import MixedElement
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.differentiation import CoefficientDerivative
from ufl.constantvalue import is_true_ufl_scalar
from ufl.indexed import Indexed
from ufl.indexing import FixedIndex, MultiIndex
from ufl.tensors import as_tensor

# An exception to the rule that ufl.* does not depend on ufl.algorithms.* ...
from ufl.algorithms import compute_form_adjoint, \
                           compute_form_action, \
                           compute_energy_norm, \
                           compute_form_lhs, \
                           compute_form_rhs, \
                           compute_form_functional, \
                           expand_derivatives, \
                           as_form

# Part of the external interface
from ufl.algorithms import replace

def lhs(form):
    """UFL form operator:
    Given a combined bilinear and linear form,
    extract the left hand side (bilinear form part).

    Example:

        a = u*v*dx + f*v*dx
        a = lhs(a) -> u*v*dx
    """
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_lhs(form)

def rhs(form):
    """UFL form operator:
    Given a combined bilinear and linear form,
    extract the right hand side (negated linear form part).

    Example:

        a = u*v*dx + f*v*dx
        L = rhs(a) -> -f*v*dx
    """
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_rhs(form)

def system(form):
    "UFL form operator: Split a form into the left hand side and right hand side, see lhs and rhs."
    return lhs(form), rhs(form)

def functional(form): # TODO: Does this make sense for anything other than testing?
    "UFL form operator: Extract the functional part of form."
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_functional(form)

def action(form, coefficient=None):
    """UFL form operator:
    Given a bilinear form, return a linear form
    with an additional coefficient, representing the
    action of the form on the coefficient. This can be
    used for matrix-free methods."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_action(form, coefficient)

def energy_norm(form, coefficient=None):
    """UFL form operator:
    Given a bilinear form, return a linear form
    with an additional coefficient, representing the
    action of the form on the coefficient. This can be
    used for matrix-free methods."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_energy_norm(form, coefficient)

def adjoint(form, reordered_arguments=None):
    """UFL form operator:
    Given a combined bilinear form, compute the adjoint form by
    changing the ordering (count) of the test and trial functions.

    By default, new Argument objects will be created with
    opposite ordering. However, if the adjoint form is to
    be added to other forms later, their arguments must match.
    In that case, the user must provide a tuple reordered_arguments=(u2,v2).
    """
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_adjoint(form, reordered_arguments)

def zero_lists(shape):
    ufl_assert(len(shape) > 0, "Invalid shape.")
    if len(shape) == 1:
        return [0]*shape[0]
    else:
        return [zero_lists(shape[1:]) for i in range(shape[0])]

def set_list_item(li, i, v):
    # Get to the innermost list
    if len(i) > 1:
        for j in i[:-1]:
            li = li[j]
    # Set item in innermost list
    li[i[-1]] = v

def _handle_derivative_arguments(coefficient, argument):
    # Wrap single coefficient in tuple for uniform treatment below
    if isinstance(coefficient, (list,tuple)):
        coefficients = tuple(coefficient)
    else:
        coefficients = (coefficient,)

    if argument is None:
        # Try to create argument if not provided
        if not all(isinstance(c, Coefficient) for c in coefficients):
            error("Can only create arguments automatically for non-indexed coefficients.")

        elements = [c.element() for c in coefficients]
        if len(elements) > 1:
            elm = MixedElement(*elements)
            arguments = split(Argument(elm))
        else:
            elm, = elements
            arguments = (Argument(elm),)
    else:
        # Wrap single argument in tuple for uniform treatment below
        if isinstance(argument, (list,tuple)):
            arguments = tuple(argument)
        else:
            n = len(coefficients)
            if n == 1:
                arguments = (argument,)
            else:
                if argument.shape() == (n,):
                    arguments = tuple(argument[i] for i in range(n))
                else:
                    arguments = split(argument)

    # Build mapping from coefficient to argument
    m = {}
    for (c, a) in izip(coefficients, arguments):
        ufl_assert(c.shape() == a.shape(), "Coefficient and argument shapes do not match!")
        if isinstance(c, Coefficient):
            m[c] = a
        else:
            ufl_assert(isinstance(c, Indexed), "Invalid coefficient type for %s" % repr(c))
            f, i = c.operands()
            ufl_assert(isinstance(f, Coefficient), "Expecting an indexed coefficient, not %s" % repr(f))
            ufl_assert(isinstance(i, MultiIndex) and all(isinstance(j, FixedIndex) for j in i),
                       "Expecting one or more fixed indices, not %s" % repr(i))
            i = tuple(int(j) for j in i)
            if f not in m:
                m[f] = {}
            m[f][i] = a

    # Merge coefficient derivatives (arguments) based on indices
    for c, p in m.iteritems():
        if isinstance(p, dict):
            a = zero_lists(c.shape())
            for i, g in p.iteritems():
                set_list_item(a, i, g)
            m[c] = as_tensor(a)

    # Wrap and return generic tuples
    items = sorted(m.items(), key=lambda x: x[0].count())
    coefficients = Tuple(*[item[0] for item in items])
    arguments = Tuple(*[item[1] for item in items])
    return coefficients, arguments

def derivative(form, coefficient, argument=None, coefficient_derivatives=None):
    """UFL form operator:
    Given any form, compute the linearization of the
    form with respect to the given Coefficient.
    The resulting form has one additional Argument
    in the same finite element space as the coefficient.
    A tuple of Coefficients may be provided in place of
    a single Coefficient, in which case the new Argument
    argument is based on a MixedElement created from this tuple."""

    coefficients, arguments = _handle_derivative_arguments(coefficient, argument)

    coefficient_derivatives = coefficient_derivatives or {}

    # Got a form? Apply derivatives to the integrands in turn.
    if isinstance(form, Form):
        integrals = []
        for itg in form.integrals():
            fd = CoefficientDerivative(itg.integrand(), coefficients,
                                       arguments, coefficient_derivatives)
            integrals.append(itg.reconstruct(fd))
        return Form(integrals)

    elif isinstance(form, Expr):
        # What we got was in fact an integrand
        return CoefficientDerivative(form, coefficients, arguments,
                                     coefficient_derivatives)

    error("Invalid argument type %s." % str(type(form)))

def sensitivity_rhs(a, u, L, v):
    """UFL form operator:
    Compute the right hand side for a sensitivity calculation system.

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
