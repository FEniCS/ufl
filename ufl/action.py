# -*- coding: utf-8 -*-
"""This module defines the Action class."""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.form import BaseForm, FormSum, Form
from ufl.argument import Argument
from ufl.coefficient import Coefficient, Cofunction
from ufl.differentiation import CoefficientDerivative

# --- The Action class represents the action of a numerical object that needs
#     to be computed at assembly time ---


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
        "_hash")

    def __getnewargs__(self):
        return (self._left, self._right)

    def __new__(cls, *args, **kw):
        left, right = args

        # Check trivial case
        if left == 0 or right == 0:
            return 0

        if isinstance(left, FormSum):
            # Action distributes over sums on the LHS
            return FormSum(*[(Action(component, right), 1)
                             for component in left.components()])
        if isinstance(right, FormSum):
            # Action also distributes over sums on the RHS
            return FormSum(*[(Action(left, component), 1)
                             for component in right.components()])

        return super(Action, cls).__new__(cls)

    def __init__(self, left, right):
        BaseForm.__init__(self)

        self._left = left
        self._right = right
        self.ufl_operands = (self._left, self._right)

        if isinstance(right, CoefficientDerivative):
            # Action differentiation pushes differentiation through
            # right as a consequence of Leibniz formula.
            right, *_ = right.ufl_operands

        if isinstance(right, (Form, Action)):
            if (left.arguments()[-1].ufl_function_space().dual()
                != right.arguments()[0].ufl_function_space()):

                raise TypeError("Incompatible function spaces in Action")
        elif isinstance(right, (Coefficient, Cofunction, Argument)):
            if (left.arguments()[-1].ufl_function_space()
                != right.ufl_function_space()):

                raise TypeError("Incompatible function spaces in Action")
        else:
            raise TypeError("Incompatible argument in Action: %s" % type(right))

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

        if isinstance(self._right, BaseForm):
            self._arguments = self._left.arguments()[:-1] \
                + self._right.arguments()[1:]
        elif isinstance(self._right, Coefficient):
            self._arguments = self._left.arguments()[:-1]
        else:
            raise TypeError

    def equals(self, other):
        if not isinstance(other, Action):
            return False
        if self is other:
            return True
        return (self._left == other._left and self._right == other._right)

    def __str__(self):
        return "Action(%s, %s)" % (repr(self._left), repr(self._right))

    def __repr__(self):
        return self._repr

    def __hash__(self):
        "Hash code for use in dicts "
        if self._hash is None:
            self._hash = hash(tuple(["Action",
                                     hash(self._right),
                                     hash(self._left)]))
        return self._hash
