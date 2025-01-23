"""This module defines the Action class."""
# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022.

from itertools import chain

from ufl import matrix  # noqa 401
from ufl.algebra import Sum
from ufl.argument import Argument, Coargument
from ufl.coefficient import BaseCoefficient, Coefficient
from ufl.constantvalue import Zero
from ufl.core.ufl_type import ufl_type
from ufl.differentiation import CoefficientDerivative
from ufl.form import BaseForm, Form, FormSum, ZeroBaseForm

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
        "_arguments",
        "_coefficients",
        "_domains",
        "_hash",
        "_left",
        "_repr",
        "_right",
        "ufl_operands",
    )

    def __new__(cls, *args, **kw):
        """Create a new Action."""
        left, right = args

        # Check trivial case
        if left == 0 or right == 0:
            if isinstance(left, Zero):
                # There is no point in checking the action arguments
                # if `left` is a `ufl.Zero` as those objects don't have arguments.
                # We can also not reliably determine the `ZeroBaseForm` arguments.
                return ZeroBaseForm(())
            # Still need to work out the ZeroBaseForm arguments.
            new_arguments, _ = _get_action_form_arguments(left, right)
            return ZeroBaseForm(new_arguments)

        # Coarguments (resp. Argument) from V* to V* (resp. from V to V) are identity matrices,
        # i.e. we have: V* x V -> R (resp. V x V* -> R).
        if isinstance(left, (Coargument, Argument)):
            return right
        if isinstance(right, (Coargument, Argument)):
            return left

        # Action distributes over sums on the LHS
        if isinstance(left, Sum):
            return FormSum(*((Action(component, right), 1) for component in left.ufl_operands))
        elif isinstance(left, FormSum):
            return FormSum(
                *((Action(c, right), w) for c, w in zip(left.components(), left.weights()))
            )

        # Action also distributes over sums on the RHS
        if isinstance(right, Sum):
            return FormSum(*((Action(left, component), 1) for component in right.ufl_operands))
        elif isinstance(right, FormSum):
            return FormSum(
                *((Action(left, c), w) for c, w in zip(right.components(), right.weights()))
            )

        return super(Action, cls).__new__(cls)

    def __init__(self, left, right):
        """Initialise."""
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
        """Get the tuple of function spaces of the underlying form."""
        if isinstance(self._right, Form):
            return self._left.ufl_function_spaces()[:-1] + self._right.ufl_function_spaces()[1:]
        elif isinstance(self._right, Coefficient):
            return self._left.ufl_function_spaces()[:-1]

    def left(self):
        """Get left."""
        return self._left

    def right(self):
        """Get right."""
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

        # Collect domains
        self._domains = join_domains(
            chain.from_iterable(e.ufl_domains() for e in self.ufl_operands)
        )

    def ufl_domains(self):
        """Return all domains found in the base form."""
        if self._domains is None:
            self._analyze_domains()
        return self._domains

    def equals(self, other):
        """Check if two Actions are equal."""
        if type(other) is not Action:
            return False
        if self is other:
            return True
        # Make sure we are returning a boolean as left and right equalities can be `ufl.Equation`s
        # if the underlying objects are `ufl.BaseForm`.
        return bool(self._left == other._left) and bool(self._right == other._right)

    def __str__(self):
        """Format as a string."""
        return f"Action({self._left}, {self._right})"

    def __repr__(self):
        """Representation."""
        return self._repr

    def __hash__(self):
        """Hash."""
        if self._hash is None:
            self._hash = hash(("Action", hash(self._right), hash(self._left)))
        return self._hash


def _check_function_spaces(left, right):
    """Check if the function spaces of left and right match."""
    # Action differentiation pushes differentiation through
    # right as a consequence of Leibniz formula.
    if isinstance(right, CoefficientDerivative):
        right, *_ = right.ufl_operands
    if isinstance(left, CoefficientDerivative):
        left, *_ = left.ufl_operands

    # `Zero` doesn't contain any information about the function space.
    # -> Not a problem since Action will get simplified with a
    #    `ZeroBaseForm` which won't take into account the arguments on
    #    the right because of argument contraction.
    # This occurs for:
    # `derivative(Action(A, B), u)` with B is an `Expr` such that dB/du == 0
    # -> `derivative(B, u)` becomes `Zero` when expanding derivatives since B is an Expr.
    if isinstance(left, Zero) or isinstance(right, Zero):
        return

    # `left` can also be a Coefficient in V (= V**), e.g.
    # `action(Coefficient(V), Cofunction(V.dual()))`.
    if isinstance(left, Coefficient):
        V_left = left.ufl_function_space()
    elif isinstance(left, BaseForm):
        V_left = left.arguments()[-1].ufl_function_space().dual()
    else:
        raise TypeError("Action left argument must be either Coefficient or BaseForm")
    if isinstance(right, Coefficient):
        V_right = right.ufl_function_space()
    elif isinstance(right, BaseForm):
        V_right = right.arguments()[0].ufl_function_space().dual()
    else:
        raise TypeError("Action right argument must be either Coefficient or BaseForm")

    if V_left.dual() != V_right:
        raise TypeError("Incompatible function spaces in Action")


def _get_action_form_arguments(left, right):
    """Perform argument contraction to work out the arguments of Action."""
    coefficients = ()
    # `left` can also be a Coefficient in V (= V**), e.g.
    # `action(Coefficient(V), Cofunction(V.dual()))`.
    left_args = left.arguments()[:-1] if not isinstance(left, Coefficient) else ()
    if isinstance(right, BaseForm):
        arguments = left_args + right.arguments()[1:]
        coefficients += right.coefficients()
    elif isinstance(right, CoefficientDerivative):
        # Action differentiation pushes differentiation through
        # right as a consequence of Leibniz formula.
        from ufl.algorithms.analysis import extract_terminals_with_domain

        right_args, right_coeffs, _ = extract_terminals_with_domain(right)
        arguments = left_args + tuple(right_args)
        coefficients += tuple(right_coeffs)
    elif isinstance(right, (BaseCoefficient, Zero)):
        arguments = left_args
        # When right is ufl.Zero, Action gets simplified so updating
        # coefficients here doesn't matter
        coefficients += (right,)
    elif isinstance(right, Argument):
        arguments = left_args + (right,)
    else:
        raise TypeError

    if isinstance(left, BaseForm):
        coefficients += left.coefficients()

    return arguments, coefficients
