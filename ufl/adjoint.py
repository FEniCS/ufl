# -*- coding: utf-8 -*-
"""This module defines the Matrix class."""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from ufl.form import BaseForm, FormSum
# --- The Adjoint class represents the adjoint of a numerical object that needs to be computed at compile time ---


class Adjoint(BaseForm):
    """UFL base form type: respresents the adjoint of an object"""

    __slots__ = (
        "_form",
        "_repr",
        "_arguments")
    _globalcount = 0

    def __getnewargs__(self):
        return (self._form)

    def __new__(cls, *args, **kw):
        form = args[0]
        if isinstance(form, FormSum):
            # Adjoint distributes over sums
            return FormSum(*[(Adjoint(component), 1) for component in form.components()])

        return super(Adjoint, cls).__new__(cls)

    def __init__(self, form):
        BaseForm.__init__(self)

        self._form = form

        self._repr = "Adjoint(%s)" % repr(self._form)

    def ufl_function_spaces(self):
        "Get the tuple of function spaces of the underlying form"
        return self._form.ufl_function_spaces()

    def _analyze_form_arguments(self):
        "Define arguments of a adjoint of a form as the reverse of the form arguments"
        self._arguments = self._form.arguments[::-1]

    def __str__(self):
        return "Adjoint(%s)" % self._form

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Adjoint):
            return False
        if self is other:
            return True
        return (self._form == other._form)
