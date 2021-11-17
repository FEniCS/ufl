# -*- coding: utf-8 -*-
"""This module defines the Interp class."""

# Copyright (C) 2019 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021

from ufl.core.ufl_type import ufl_type
from ufl.coefficient import Coefficient
from ufl.core.base_form_operator import BaseFormOperator


@ufl_type(num_ops="varying", inherit_indices_from_operand=0, is_differential=True)
class Interp(BaseFormOperator):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, operand, function_space, derivatives=None, result_coefficient=None, argument_slots=()):
        r"""
        :arg expr: a UFL expression to interpolate.
        :arg function_space: the :class:`.FunctionSpace` to interpolate into (or else
        an existing :class:`.Coefficient`).
        :param derivatives: tuple specifiying the derivative multiindex.
        :param result_coefficient: ufl.Coefficient associated to Interp representing what is produced by the operator
        :param argument_slots: tuple composed containing expressions with ufl.Argument or ufl.Coefficient objects.
        """

        if isinstance(function_space, Coefficient):
            # Is there anything else we should do with this Coefficient?
            function_space = function_space.ufl_function_space()

        BaseFormOperator.__init__(self, operand, function_space=function_space, derivatives=derivatives,
                                  result_coefficient=result_coefficient, argument_slots=argument_slots)

    def __repr__(self):
        "Default repr string construction for external operators."
        r = "Interp(%s; %s; %s; derivatives=%s)" % (", ".join(repr(op) for op in self.ufl_operands),
                                                    repr(self.ufl_function_space()),
                                                    ", ".join(repr(arg) for arg in self.argument_slots()),
                                                    repr(self.derivatives))
        return r

    def __eq__(self, other):
        if not isinstance(other, Interp):
            return False
        if self is other:
            return True
        return (type(self) == type(other) and
                # Operands' output spaces will be taken into account via Interp.__eq__
                # -> N(Interp(u, V1); v*) and N(Interp(u, V2); v*) will compare different.
                all(a == b for a, b in zip(self.ufl_operands, other.ufl_operands)) and
                all(a == b for a, b in zip(self._argument_slots, other._argument_slots)) and
                self.derivatives == other.derivatives and
                self.ufl_function_space() == other.ufl_function_space())
