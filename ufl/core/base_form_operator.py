
# -*- coding: utf-8 -*-
"""This module defines the BaseFormOperator class, which is the base class for objects that can be seen as forms and as operators such as ExternalOperator or Interp."""

# Copyright (C) 2019 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022

from ufl.coefficient import Coefficient
from ufl.argument import Argument
from ufl.core.operator import Operator
from ufl.form import BaseForm
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import as_ufl
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace
from ufl.referencevalue import ReferenceValue


@ufl_type(num_ops="varying", is_differential=True)
class BaseFormOperator(Operator, BaseForm):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=()):
        r"""
        :param operands: operands on which acts the operator.
        :param function_space: the :class:`.FunctionSpace`,
               or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param result_coefficient: ufl.Coefficient associated to the operator representing what is produced by the operator
        :param argument_slots: tuple composed containing expressions with ufl.Argument or ufl.Coefficient objects.
        """

        BaseForm.__init__(self)
        ufl_operands = tuple(map(as_ufl, operands))
        argument_slots = tuple(map(as_ufl, argument_slots))
        Operator.__init__(self, ufl_operands)

        # -- Function space -- #
        if isinstance(function_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map
            # the cell to The Default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, AbstractFunctionSpace):
            raise ValueError("Expecting a FunctionSpace or FiniteElement.")

        # -- Derivatives -- #
        # Some BaseFormOperator does have derivatives (e.g. ExternalOperator)
        # while other don't since they are fully determined by their
        # argument slots (e.g. Interp)
        self.derivatives = derivatives

        # Produce the resulting Coefficient: Is that really needed?
        if result_coefficient is None:
            result_coefficient = Coefficient(function_space)
        elif not isinstance(result_coefficient, (Coefficient, ReferenceValue)):
            raise TypeError('Expecting a Coefficient and not %s', type(result_coefficient))
        self._result_coefficient = result_coefficient

        # -- Argument slots -- #
        if len(argument_slots) == 0:
            # Make v*
            v_star = Argument(function_space.dual(), 0)
            argument_slots = (v_star,)
        self._argument_slots = argument_slots

        # Internal variables for caching coefficient data
        self._coefficients = None

    # BaseFormOperators don't have free indices.
    ufl_free_indices = ()
    ufl_index_dimensions = ()

    def result_coefficient(self, unpack_reference=True):
        "Returns the coefficient produced by the base form operator"
        result_coefficient = self._result_coefficient
        if unpack_reference and isinstance(result_coefficient, ReferenceValue):
            return result_coefficient.ufl_operands[0]
        return result_coefficient

    def argument_slots(self, outer_form=False):
        r"""Returns a tuple of expressions containing argument and coefficient based expressions.
            We get an argument uhat when we take the Gateaux derivative in the direction uhat:
                -> d/du N(u; v*) = dNdu(u; uhat, v*) where uhat is a ufl.Argument and v* a ufl.Coargument
            Applying the action replace the last argument by coefficient:
                -> action(dNdu(u; uhat, v*), w) = dNdu(u; w, v*) where du is a ufl.Coefficient
        """
        if not outer_form:
            return self._argument_slots
        # Takes into account argument contraction when an external operator is in an outer form:
        # For example:
        #   F = N(u; v*) * v * dx can be seen as Action(v1 * v * dx, N(u; v*))
        #   => F.arguments() should return (v,)!
        from ufl.algorithms.analysis import extract_arguments
        return tuple(a for a in self._argument_slots[1:] if len(extract_arguments(a)) != 0)

    def coefficients(self):
        "Return all ``BaseCoefficient`` objects found in base form operator."
        if self._coefficients is None:
            self._analyze_form_arguments()
        return self._coefficients

    def _analyze_form_arguments(self):
        "Analyze which Argument and Coefficient objects can be found in the base form."
        from ufl.algorithms.analysis import extract_arguments, extract_coefficients
        arguments = tuple(a for arg in self.argument_slots() for a in extract_arguments(arg))
        coefficients = tuple(c for op in self.ufl_operands for c in extract_coefficients(op))
        # Define canonical numbering of arguments and coefficients
        from collections import OrderedDict
        # 1) Need concept of order since we may have arguments with the same number
        #    because of form composition (`argument_slots(outer_form=True)`):
        #    Example: Let u \in V1 and N \in V2 and F = N(u; v*) * dx, then
        #    `derivative(F, u)` will contain dNdu(u; uhat, v*) with v* = Argument(0, V2)
        #    and uhat = Argument(0, V1) (since F.arguments() = ())
        # 2) Having sorted arguments also makes BaseFormOperator compatible with other
        #    BaseForm objects for which the highest-numbered argument always comes last.
        self._arguments = tuple(sorted(OrderedDict.fromkeys(arguments), key=lambda x: x.number()))
        self._coefficients = tuple(sorted(set(coefficients), key=lambda x: x.count()))

    def count(self):
        "Returns the count associated to the coefficient produced by the external operator"
        return self._count

    @property
    def _count(self):
        return self.result_coefficient()._count

    @property
    def ufl_shape(self):
        "Returns the UFL shape of the coefficient.produced by the operator"
        return self.result_coefficient()._ufl_shape

    def ufl_function_space(self):
        "Returns the ufl function space associated to the operator"
        return self.result_coefficient()._ufl_function_space

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None,
                               result_coefficient=None, argument_slots=None):
        "Return a new object of the same type with new operands."
        deriv_multiindex = derivatives or self.derivatives

        if deriv_multiindex != self.derivatives:
            # If we are constructing a derivative
            corresponding_coefficient = None
        else:
            corresponding_coefficient = result_coefficient or self._result_coefficient

        return type(self)(*operands, function_space=function_space or self.ufl_function_space(),
                          derivatives=deriv_multiindex,
                          result_coefficient=corresponding_coefficient,
                          argument_slots=argument_slots or self.argument_slots())

    def __repr__(self):
        "Default repr string construction for external operators."
        r = "%s(%s; %s; %s; derivatives=%s)" % (type(self).__name__,
                                                ", ".join(repr(op) for op in self.ufl_operands),
                                                repr(self.ufl_function_space()),
                                                ", ".join(repr(arg) for arg in self.argument_slots()),
                                                repr(self.derivatives))
        return r

    def __hash__(self):
        "Hash code for use in dicts."
        hashdata = (type(self),
                    tuple(hash(op) for op in self.ufl_operands),
                    tuple(hash(arg) for arg in self._argument_slots),
                    self.derivatives,
                    hash(self.ufl_function_space()))
        return hash(hashdata)

    def __eq__(self, other):
        if not isinstance(other, BaseFormOperator):
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
