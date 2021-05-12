# -*- coding: utf-8 -*-
"""This module defines the Matrix class."""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.form import BaseForm, FormSum

# --- The Action class represents the adjoint of a numerical object that needs to be computed at compile time ---


class Action(BaseForm):
    """UFL base form type: respresents the adjoint of an object"""

    __slots__ = (
        "_left",
        "_right",
        "_repr",
        "_arguments",
        "_external_operators")
    _globalcount = 0

    def __getnewargs__(self):
        return (self._left, self._right)

    def __new__(cls, *args, **kw):
        assert(len(args) == 2)
        left = args[0]
        right = args[1]
        if isinstance(left, FormSum):
            # Adjoint distributes over sums
            return FormSum(*[(Action(component, right), 1) for component in left.components()])

        return super(Action, cls).__new__(cls)

    def __init__(self, left, right):
        BaseForm.__init__(self)

        self._left = left
        self._right = right

        assert(self._left.arguments()[-1].ufl_function_space() == self._right.arguments()[0].ufl_function_space())

        self._repr = "Action(%s, %s)" % (repr(self._left), repr(self._right))

    def ufl_function_spaces(self):
        "Get the tuple of function spaces of the underlying form"
        return self._form.ufl_function_spaces()

    def _analyze_form_arguments(self):
        "Define arguments of a adjoint of a form as the reverse of the form arguments"
        self._arguments = self._left.arguments()[:-1] + self._right.arguments()[1:]

    def _analyze_external_operators(self):
        "Define external_operators of Action"
        self._external_operators = tuple(set(self._left.external_operators() + self._right.external_operators()))

    def __str__(self):
        return "Action(%s, %s)" % (repr(self._left), repr(self._right))

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        if self is other:
            return True
        return (self._left == other._left)
