"""Various high level ways to transform a complete Form into a new Form."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009
# Modified by Massimiliano Leoni, 2016
# Modified by Cecile Daversin-Catty, 2018

from ufl.action import Action
from ufl.adjoint import Adjoint
from ufl.algorithms import replace  # noqa: F401
from ufl.algorithms import (compute_energy_norm, compute_form_action, compute_form_adjoint, compute_form_functional,
                            compute_form_lhs, compute_form_rhs, expand_derivatives, extract_arguments, formsplitter)
from ufl.argument import Argument
from ufl.coefficient import Coefficient, Cofunction
from ufl.constantvalue import as_ufl, is_true_ufl_scalar
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.expr import Expr, ufl_err_str
from ufl.core.multiindex import FixedIndex, MultiIndex
from ufl.differentiation import (BaseFormCoordinateDerivative, BaseFormDerivative, BaseFormOperatorCoordinateDerivative,
                                 BaseFormOperatorDerivative, CoefficientDerivative, CoordinateDerivative)
from ufl.exprcontainers import ExprList, ExprMapping
from ufl.finiteelement import MixedElement
from ufl.form import BaseForm, Form, FormSum, ZeroBaseForm, as_form
from ufl.functionspace import FunctionSpace
from ufl.geometry import SpatialCoordinate
from ufl.indexed import Indexed
from ufl.sorting import sorted_expr
from ufl.split_functions import split
from ufl.tensors import ListTensor, as_tensor
from ufl.variable import Variable


def extract_blocks(form, i=None, j=None):
    """Extract blocks.

    Given a linear or bilinear form on a mixed space,
    extract the block corresponding to the indices ix, iy.

    Example:
       a = inner(grad(u), grad(v))*dx + div(u)*q*dx + div(v)*p*dx
       extract_blocks(a, 0, 0) -> inner(grad(u), grad(v))*dx
       extract_blocks(a) -> [inner(grad(u), grad(v))*dx, div(v)*p*dx, div(u)*q*dx, 0]
    """
    return formsplitter.extract_blocks(form, i, j)


def lhs(form):
    """Get the left hand side.

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
    """Get the right hand side.

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
    """Split a form into the left hand side and right hand side.

    See ``lhs`` and ``rhs``.
    """
    return lhs(form), rhs(form)


def functional(form):  # TODO: Does this make sense for anything other than testing?
    """Extract the functional part of form."""
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_form_functional(form)


def action(form, coefficient=None, derivatives_expanded=None):
    """Get the action.

    Given a bilinear form, return a linear form
    with an additional coefficient, representing the
    action of the form on the coefficient. This can be
    used for matrix-free methods.
    For formbase objects,coefficient can be any object of the correct type,
    and this function returns an Action object.

    When `action` is being called multiple times on the same form, expanding derivatives
    become expensive -> `derivatives_expanded` enables to use caching mechanisms to avoid that.
    """
    form = as_form(form)
    is_coefficient_valid = (not isinstance(coefficient, BaseForm) or
                            (isinstance(coefficient, BaseFormOperator) and len(coefficient.arguments()) == 1))
    # Can't expand derivatives on objects that are not Form or Expr (e.g. Matrix)
    if isinstance(form, (Form, BaseFormOperator)) and is_coefficient_valid:
        if not derivatives_expanded:
            # For external operators differentiation may turn a Form into a FormSum
            form = expand_derivatives(form)
        if isinstance(form, Form):
            return compute_form_action(form, coefficient)
    return Action(form, coefficient)


def energy_norm(form, coefficient=None):
    """Get the energy norm.

    Given a bilinear form *a* and a coefficient *f*,
    return the functional :math:`a(f,f)`.
    """
    form = as_form(form)
    form = expand_derivatives(form)
    return compute_energy_norm(form, coefficient)


def adjoint(form, reordered_arguments=None, derivatives_expanded=None):
    """Get the adjoint.

    Given a combined bilinear form, compute the adjoint form by
    changing the ordering (count) of the test and trial functions, and
    taking the complex conjugate of the result.

    By default, new ``Argument`` objects will be created with
    opposite ordering. However, if the adjoint form is to
    be added to other forms later, their arguments must match.
    In that case, the user must provide a tuple *reordered_arguments*=(u2,v2).

    If the form is a baseform instance instead of a Form object, we return an Adjoint
    object instructing the adjoint to be computed at a later point.

    When `adjoint` is being called multiple times on the same form, expanding derivatives
    become expensive -> `derivatives_expanded` enables to use caching mechanisms to avoid that.
    """
    form = as_form(form)
    if isinstance(form, BaseForm):
        # Allow BaseForm objects that are not BaseForm such as Adjoint since there are cases
        # where we need to expand derivatives: e.g. to get the number of arguments
        #   => For example: Adjoint(Action(2-form, derivative(u,u)))
        if not derivatives_expanded:
            # For external operators differentiation may turn a Form into a FormSum
            form = expand_derivatives(form)
        if isinstance(form, Form):
            return compute_form_adjoint(form, reordered_arguments)
    return Adjoint(form)


def zero_lists(shape):
    """Createa list of zeros of the given shape."""
    if len(shape) == 0:
        raise ValueError("Invalid shape.")
    elif len(shape) == 1:
        return [0] * shape[0]
    else:
        return [zero_lists(shape[1:]) for i in range(shape[0])]


def set_list_item(li, i, v):
    """Set an item in a nested list."""
    # Get to the innermost list
    if len(i) > 1:
        for j in i[:-1]:
            li = li[j]
    # Set item in innermost list
    li[i[-1]] = v


def _handle_derivative_arguments(form, coefficient, argument):
    """Handle derivative arguments."""
    # Wrap single coefficient in tuple for uniform treatment below
    if isinstance(coefficient, (list, tuple, ListTensor)):
        coefficients = tuple(coefficient)
    else:
        coefficients = (coefficient,)

    if argument is None:
        # Try to create argument if not provided
        if not all(isinstance(c, (Coefficient, Cofunction, BaseFormOperator)) for c in coefficients):
            raise ValueError("Can only create arguments automatically for non-indexed coefficients.")

        # Get existing arguments from form and position the new one
        # with the next argument number
        if isinstance(form, Form):
            form_arguments = form.arguments()
        else:
            # To handle derivative(expression), which is at least used
            # in tests. Remove?
            form_arguments = extract_arguments(form)

        numbers = sorted(set(arg.number() for arg in form_arguments))
        number = max(numbers + [-1]) + 1

        # Don't know what to do with parts, let the user sort it out
        # in that case
        parts = set(arg.part() for arg in form_arguments)
        if len(parts - {None}) != 0:
            raise ValueError("Not expecting parts here, provide your own arguments.")
        part = None

        # Create argument and split it if in a mixed space
        function_spaces = [c.ufl_function_space() for c in coefficients]
        if len(function_spaces) == 1:
            arguments = (Argument(function_spaces[0], number, part),)
        else:
            # Create in mixed space over assumed (for now) same domain
            domains = [fs.ufl_domain() for fs in function_spaces]
            elements = [fs.ufl_element() for fs in function_spaces]
            assert all(fs.ufl_domain() == domains[0] for fs in function_spaces)
            elm = MixedElement(elements)
            fs = FunctionSpace(domains[0], elm)
            arguments = split(Argument(fs, number, part))
    else:
        # Wrap single argument in tuple for uniform treatment below
        if isinstance(argument, (list, tuple)):
            arguments = tuple(argument)
        else:
            n = len(coefficients)
            if n == 1:
                arguments = (argument,)
            else:
                if argument.ufl_shape == (n,):
                    arguments = tuple(argument[i] for i in range(n))
                else:
                    arguments = split(argument)

    # Build mapping from coefficient to argument
    m = {}
    for (c, a) in zip(coefficients, arguments):
        if c.ufl_shape != a.ufl_shape:
            raise ValueError("Coefficient and argument shapes do not match!")
        if isinstance(c, (Coefficient, Cofunction, BaseFormOperator, SpatialCoordinate)):
            m[c] = a
        else:
            if not isinstance(c, Indexed):
                raise ValueError(f"Invalid coefficient type for {ufl_err_str(c)}")
            f, i = c.ufl_operands
            if not isinstance(f, Coefficient):
                raise ValueError(f"Expecting an indexed coefficient, not {ufl_err_str(f)}")
            if not (isinstance(i, MultiIndex) and all(isinstance(j, FixedIndex) for j in i)):
                raise ValueError(f"Expecting one or more fixed indices, not {ufl_err_str(i)}")
            i = tuple(int(j) for j in i)
            if f not in m:
                m[f] = {}
            m[f][i] = a

    # Merge coefficient derivatives (arguments) based on indices
    for c, p in m.items():
        if isinstance(p, dict):
            a = zero_lists(c.ufl_shape)
            for i, g in p.items():
                set_list_item(a, i, g)
            m[c] = as_tensor(a)

    # Wrap and return generic tuples
    items = sorted(m.items(), key=lambda x: x[0].count())
    coefficients = ExprList(*[item[0] for item in items])
    arguments = ExprList(*[item[1] for item in items])
    return coefficients, arguments


def derivative(form, coefficient, argument=None, coefficient_derivatives=None):
    """Compute the Gateaux derivative of *form* w.r.t. *coefficient* in direction of *argument*.

    If the argument is omitted, a new ``Argument`` is created
    in the same space as the coefficient, with argument number
    one higher than the highest one in the form.

    The resulting form has one additional ``Argument``
    in the same finite element space as the coefficient.

    A tuple of ``Coefficient`` s may be provided in place of
    a single ``Coefficient``, in which case the new ``Argument``
    argument is based on a ``MixedElement`` created from this tuple.

    An indexed ``Coefficient`` from a mixed space may be provided,
    in which case the argument should be in the corresponding
    subspace of the coefficient space.

    If provided, *coefficient_derivatives* should be a mapping from
    ``Coefficient`` instances to their derivatives w.r.t. *coefficient*.
    """
    if isinstance(form, FormSum):
        # Distribute derivative over FormSum components
        return FormSum(*[(derivative(component, coefficient, argument, coefficient_derivatives), 1)
                         for component in form.components()])
    elif isinstance(form, Adjoint):
        # Is `derivative(Adjoint(A), ...)` with A a 2-form even legal ?
        # -> If yes, what's the right thing to do here ?
        raise NotImplementedError('Adjoint derivative is not supported.')
    elif isinstance(form, Action):
        # Push derivative through Action slots
        left, right = form.ufl_operands
        # Eagerly simplify spatial derivatives when Action results in a scalar.
        if not len(form.arguments()) and isinstance(coefficient, SpatialCoordinate):
            return ZeroBaseForm(())

        if len(left.arguments()) == 1:
            dleft = derivative(left, coefficient, argument, coefficient_derivatives)
            dright = derivative(right, coefficient, argument, coefficient_derivatives)
            # Leibniz formula
            return action(adjoint(dleft), right) + action(left, dright)
        else:
            raise NotImplementedError('Action derivative not supported when the left argument is not a 1-form.')

    coefficients, arguments = _handle_derivative_arguments(form, coefficient,
                                                           argument)
    if coefficient_derivatives is None:
        coefficient_derivatives = ExprMapping()
    else:
        cd = []
        for k in sorted_expr(coefficient_derivatives.keys()):
            cd += [as_ufl(k), as_ufl(coefficient_derivatives[k])]
        coefficient_derivatives = ExprMapping(*cd)

    # Got a form? Apply derivatives to the integrands in turn.
    if isinstance(form, Form):
        integrals = []
        for itg in form.integrals():
            if isinstance(coefficient, SpatialCoordinate):
                fd = CoordinateDerivative(itg.integrand(), coefficients,
                                          arguments, coefficient_derivatives)
            elif isinstance(coefficient, BaseForm) and not isinstance(coefficient, BaseFormOperator):
                # Make the `ZeroBaseForm` arguments
                arguments = form.arguments() + coefficient.arguments()
                return ZeroBaseForm(arguments)
            else:
                fd = CoefficientDerivative(itg.integrand(), coefficients,
                                           arguments, coefficient_derivatives)
            integrals.append(itg.reconstruct(fd))
        return Form(integrals)

    elif isinstance(form, BaseFormOperator):
        if not isinstance(coefficient, SpatialCoordinate):
            return BaseFormOperatorDerivative(form, coefficients, arguments, coefficient_derivatives)
        else:
            return BaseFormOperatorCoordinateDerivative(form, coefficients, arguments, coefficient_derivatives)

    elif isinstance(form, BaseForm):
        if not isinstance(coefficient, SpatialCoordinate):
            return BaseFormDerivative(form, coefficients, arguments, coefficient_derivatives)
        else:
            return BaseFormCoordinateDerivative(form, coefficients, arguments, coefficient_derivatives)

    elif isinstance(form, Expr):
        # What we got was in fact an integrand
        if not isinstance(coefficient, SpatialCoordinate):
            return CoefficientDerivative(form, coefficients,
                                         arguments, coefficient_derivatives)
        else:
            return CoordinateDerivative(form, coefficients,
                                        arguments, coefficient_derivatives)

    raise ValueError(f"Invalid argument type {type(form)}.")


def sensitivity_rhs(a, u, L, v):
    r"""Compute the right hand side for a sensitivity calculation system.

    The derivation behind this computation is as follows.
    Assume *a*, *L* to be bilinear and linear forms
    corresponding to the assembled linear system

    .. math::

        Ax = b.

    Where *x* is the vector of the discrete function corresponding to *u*.
    Let *v* be some scalar variable this equation depends on.
    Then we can write

    .. math::
        0 = \\frac{d}{dv}(Ax-b) = \\frac{dA}{dv} x + A \\frac{dx}{dv} -
        \\frac{db}{dv},

        A \\frac{dx}{dv} = \\frac{db}{dv} - \\frac{dA}{dv} x,

    and solve this system for :math:`\\frac{dx}{dv}`, using the same bilinear
    form *a* and matrix *A* from the original system.
    Assume the forms are written
    ::

        v = variable(v_expression)
        L = IL(v)*dx
        a = Ia(v)*dx

    where ``IL`` and ``Ia`` are integrand expressions.
    Define a ``Coefficient u`` representing the solution
    to the equations. Then we can compute :math:`\\frac{db}{dv}`
    and :math:`\\frac{dA}{dv}` from the forms
    ::

        da = diff(a, v)
        dL = diff(L, v)

    and the action of ``da`` on ``u`` by
    ::

        dau = action(da, u)

    In total, we can build the right hand side of the system
    to compute :math:`\\frac{du}{dv}` with the single line
    ::

        dL = diff(L, v) - action(diff(a, v), u)

    or, using this function,
    ::

        dL = sensitivity_rhs(a, u, L, v)
    """
    if not (isinstance(a, Form) and isinstance(u, Coefficient) and isinstance(L, Form) and isinstance(v, Variable)):
        raise ValueError("Expecting (a, u, L, v), (bilinear form, function, linear form and scalar variable).")
    if not is_true_ufl_scalar(v):
        raise ValueError("Expecting scalar variable.")
    from ufl.operators import diff
    return diff(L, v) - action(diff(a, v), u)
