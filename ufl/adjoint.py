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
        "_arguments",
        "_external_operators",
        "_hash")
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
        self._hash = None
        self._repr = "Adjoint(%s)" % repr(self._form)

    def ufl_function_spaces(self):
        "Get the tuple of function spaces of the underlying form"
        return self._form.ufl_function_spaces()

    def form(self):
        return self._form

    def _analyze_form_arguments(self):
        "Define arguments of a adjoint of a form as the reverse of the form arguments"
        self._arguments = self._form.arguments()[::-1]

    def _analyze_external_operators(self):
        "Define external_operators of Adjoint"
        self._external_operators = self._form.external_operators()

    def __str__(self):
        return "Adjoint(%s)" % self._form

    def __repr__(self):
        return self._repr

    def __hash__(self):
        "Hash code for use in dicts (includes incidental numbering of indices etc.)"
        if self._hash is None:
            self._hash = hash(tuple(["Adjoint", hash(self._form)]))
        return self._hash
