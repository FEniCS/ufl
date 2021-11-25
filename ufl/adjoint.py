# -*- coding: utf-8 -*-
"""This module defines the Adjoint class."""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from ufl.form import BaseForm, FormSum
# --- The Adjoint class represents the adjoint of a numerical object that
#     needs to be computed at assembly time ---


class Adjoint(BaseForm):
    """UFL base form type: represents the adjoint of an object.

    Adjoint objects will result when the adjoint of an assembled object
    (e.g. a Matrix) is taken. This delays the evaluation of the adjoint until
    assembly occurs.
    """

    __slots__ = (
        "_form",
        "_repr",
        "_arguments",
        "ufl_operands",
        "_hash")

    def __getnewargs__(self):
        return (self._form)

    def __new__(cls, *args, **kw):
        form = args[0]
        # Check trivial case
        if form == 0:
            return 0

        if isinstance(form, Adjoint):
            return form._form
        elif isinstance(form, FormSum):
            # Adjoint distributes over sums
            return FormSum(*[(Adjoint(component), 1)
                             for component in form.components()])

        return super(Adjoint, cls).__new__(cls)

    def __init__(self, form):
        BaseForm.__init__(self)

        if len(form.arguments()) != 2:
            raise ValueError("Can only take Adjoint of a 2-form.")

        self._form = form
        self.ufl_operands = (self._form,)
        self._hash = None
        self._repr = "Adjoint(%s)" % repr(self._form)

    def ufl_function_spaces(self):
        "Get the tuple of function spaces of the underlying form"
        return self._form.ufl_function_spaces()

    def form(self):
        return self._form

    def _analyze_form_arguments(self):
        """The arguments of adjoint are the reverse of the form arguments."""
        self._arguments = self._form.arguments()[::-1]

    def __eq__(self, other):
        if not isinstance(other, Adjoint):
            return False
        if self is other:
            return True
        return (self._form == other._form)

    def __str__(self):
        return "Adjoint(%s)" % self._form

    def __repr__(self):
        return self._repr

    def __hash__(self):
        """Hash code for use in dicts."""
        if self._hash is None:
            self._hash = hash(tuple(["Adjoint", hash(self._form)]))
        return self._hash
