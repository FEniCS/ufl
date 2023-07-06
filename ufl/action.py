# -*- coding: utf-8 -*-
"""This module defines the Action class."""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022.

from ufl.form import BaseForm, FormSum, Form, ZeroBaseForm
from ufl.core.ufl_type import ufl_type
from ufl.algebra import Sum
from ufl.constantvalue import Zero
from ufl.argument import Argument
from ufl.coefficient import BaseCoefficient, Coefficient, Cofunction
from ufl.differentiation import CoefficientDerivative
from ufl.matrix import Matrix

# --- The Action class represents the action of a numerical object that needs
#     to be computed at assembly time ---


@ufl_type()
class Action(BaseForm):
    """UFL base form type: respresents the action of an object on another.
    For example:
        res = Ax
    A would be the first argument, left and x would be the second argument,
    right.

    Action objects will result when the action of an assembled object
    (e.g. a Matrix) is taken. This delays the evaluation of the action until
    assembly occurs.
    """

    __slots__ = (
        "_left",
        "_right",
        "ufl_operands",
        "_repr",
        "_arguments",
        "_coefficients",
        "_domains",
        "_hash")

    def __getnewargs__(self):
        return (self._left, self._right)

    def __new__(cls, *args, **kw):
        left, right = args

        # Check trivial case
        if left == 0 or right == 0:
            # Check compatibility of function spaces
            _check_function_spaces(left, right)
            # Still need to work out the ZeroBaseForm arguments.
            new_arguments, _ = _get_action_form_arguments(left, right)
            return ZeroBaseForm(new_arguments)

        if isinstance(left, (FormSum, Sum)):
            # Action distributes over sums on the LHS
            return FormSum(*[(Action(component, right), 1)
                             for component in left.ufl_operands])
        if isinstance(right, (FormSum, Sum)):
            # Action also distributes over sums on the RHS
            return FormSum(*[(Action(left, component), 1)
                             for component in right.ufl_operands])

        return super(Action, cls).__new__(cls)

    def __init__(self, left, right):
        BaseForm.__init__(self)

        self._left = left
        self._right = right
        self.ufl_operands = (self._left, self._right)
        self._domains = None

        # Check compatibility of function spaces
        _check_function_spaces(left, right)

        self._repr = "Action(%s, %s)" % (repr(self._left), repr(self._right))
        self._hash = None

    def ufl_function_spaces(self):
        "Get the tuple of function spaces of the underlying form"
        if isinstance(self._right, Form):
            return self._left.ufl_function_spaces()[:-1] \
                + self._right.ufl_function_spaces()[1:]
        elif isinstance(self._right, Coefficient):
            return self._left.ufl_function_spaces()[:-1]

    def left(self):
        return self._left

    def right(self):
        return self._right

    def _analyze_form_arguments(self):
        """Compute the Arguments of this Action.

        The highest number Argument of the left operand and the lowest number
        Argument of the right operand are consumed by the action.
        """
        self._arguments, self._coefficients = _get_action_form_arguments(self._left, self._right)

    def _analyze_domains(self):
        """Analyze which domains can be found in Action."""
        from ufl.domain import join_domains
        # Collect unique domains
        self._domains = join_domains([e.ufl_domain() for e in self.ufl_operands])

    def equals(self, other):
        if type(other) is not Action:
            return False
        if self is other:
            return True
        return (self._left == other._left and self._right == other._right)

    def __str__(self):
        return "Action(%s, %s)" % (str(self._left), str(self._right))

    def __repr__(self):
        return self._repr

    def __hash__(self):
        "Hash code for use in dicts "
        if self._hash is None:
            self._hash = hash(("Action",
                               hash(self._right),
                               hash(self._left)))
        return self._hash


def _check_function_spaces(left, right):
    """Check if the function spaces of left and right match."""

    if isinstance(right, CoefficientDerivative):
        # Action differentiation pushes differentiation through
        # right as a consequence of Leibniz formula.
        right, *_ = right.ufl_operands

    if isinstance(right, (Form, Action, Matrix, ZeroBaseForm)):
        if (left.arguments()[-1].ufl_function_space().dual()
            != right.arguments()[0].ufl_function_space()):

            raise TypeError("Incompatible function spaces in Action")
    elif isinstance(right, (Coefficient, Cofunction, Argument)):
        if (left.arguments()[-1].ufl_function_space()
            != right.ufl_function_space()):

            raise TypeError("Incompatible function spaces in Action")
    # `Zero` doesn't contain any information about the function space.
    # -> Not a problem since Action will get simplified with a `ZeroBaseForm`
    #    which won't take into account the arguments on the right because of argument contraction.
    # This occurs for:
    # `derivative(Action(A, B), u)` with B is an `Expr` such that dB/du == 0
    # -> `derivative(B, u)` becomes `Zero` when expanding derivatives since B is an Expr.
    elif not isinstance(right, Zero):
        raise TypeError("Incompatible argument in Action: %s" % type(right))


def _get_action_form_arguments(left, right):
    """Perform argument contraction to work out the arguments of Action"""

    if isinstance(right, CoefficientDerivative):
        # Action differentiation pushes differentiation through
        # right as a consequence of Leibniz formula.
        right, *_ = right.ufl_operands

    coefficients = ()
    if isinstance(right, BaseForm):
        arguments = left.arguments()[:-1] + right.arguments()[1:]
        coefficients += right.coefficients()
    elif isinstance(right, (BaseCoefficient, Zero)):
        arguments = left.arguments()[:-1]
        # When right is ufl.Zero, Action gets simplified so updating
        # coefficients here doesn't matter
        coefficients += (right,)
    elif isinstance(right, Argument):
        arguments = left.arguments()[:-1] + (right,)
    else:
        raise TypeError

    if isinstance(left, BaseForm):
        coefficients += left.coefficients()

    return arguments, coefficients
