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
from ufl.constantvalue import as_ufl
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace
from ufl.argument import Argument, Coargument
from ufl.core.base_form_operator import BaseFormOperator
from ufl.duals import is_dual


@ufl_type(num_ops="varying", is_differential=True)
class Interp(BaseFormOperator):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, expr, v, result_coefficient=None):
        r""" Symbolic representation of the interpolation operator.

        :arg expr: a UFL expression to interpolate.
        :arg v: the :class:`.FunctionSpace` to interpolate into or the :class:`.Coargument`
                defined on the dual of the :class:`.FunctionSpace` to interpolate into.
        :param result_coefficient: the :class:`.Coefficient` representing what is produced by the operator
        """

        if isinstance(v, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map
            # the cell to The Default Mesh
            element = v
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
            v = Argument(function_space.dual(), 0)
        elif isinstance(v, AbstractFunctionSpace):
            if is_dual(v):
                raise ValueError('Expecting a primal function space.')
            v = Argument(v.dual(), 0)
        elif not isinstance(v, Coargument):
            raise ValueError("Expecting a Coargument, FunctionSpace or FiniteElement.")
        # If v is a Coargument, should we impose its number to be 0 ?

        expr = as_ufl(expr)
        # Reversed order convention
        argument_slots = (v, expr)
        # Get the primal space (V** = V)
        function_space = v.ufl_function_space().dual()
        # Set the operand as `expr` for DAG traversal purpose.
        operand = expr
        BaseFormOperator.__init__(self, operand, function_space=function_space,
                                  result_coefficient=result_coefficient,
                                  argument_slots=argument_slots)

    def _ufl_expr_reconstruct_(self, expr, v=None, result_coefficient=None):
        "Return a new object of the same type with new operands."
        v = v or self.argument_slots()[0]
        # This should check if we need a new coefficient, i.e. if we need
        # to pass `self._result_coefficient` when `result_coefficient` is None.
        # -> `result_coefficient` is deprecated so it shouldn't be a problem!
        result_coefficient = result_coefficient or self._result_coefficient
        return type(self)(expr, v, result_coefficient=result_coefficient)

    def __repr__(self):
        "Default repr string construction for Interp."
        r = "Interp(%s; %s)" % (", ".join(repr(arg) for arg in reversed(self.argument_slots())),
                                repr(self.ufl_function_space()))
        return r

    def __str__(self):
        "Default str string construction for Interp."
        s = "Interp(%s; %s)" % (", ".join(str(arg) for arg in reversed(self.argument_slots())),
                                str(self.ufl_function_space()))
        return s

    def __eq__(self, other):
        if not isinstance(other, Interp):
            return False
        if self is other:
            return True
        return (type(self) == type(other) and
                all(a == b for a, b in zip(self._argument_slots, other._argument_slots)) and
                self.derivatives == other.derivatives and
                self.ufl_function_space() == other.ufl_function_space())
