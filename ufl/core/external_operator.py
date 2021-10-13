# -*- coding: utf-8 -*-
"""This module defines the ``ExternalOperator`` class, which symbolically represents operators that are not straightforwardly expressible in UFL. A practical implementation is required at a later stage to define how this operator should be evaluated as well as its derivatives from a given set of operands.
"""

# Copyright (C) 2019 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021

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


@ufl_type(num_ops="varying", inherit_indices_from_operand=0, is_differential=True)
class ExternalOperator(Operator, BaseForm):

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, *operands, function_space, derivatives=None, result_coefficient=None, argument_slots=()):
        r"""
        :param operands: operands on which acts the :class:`ExternalOperator`.
        :param function_space: the :class:`.FunctionSpace`,
               or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
        :param derivatives: tuple specifiying the derivative multiindex.
        :param result_coefficient: ufl.Coefficient associated to the ExternalOperator representing what is produced by the operator
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
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                raise TypeError("Expecting a tuple for derivatives and not %s" % derivatives)
            if not len(derivatives) == len(self.ufl_operands):
                raise ValueError("Expecting a size of %s for %s" % (len(self.ufl_operands), derivatives))
            if not all(isinstance(d, int) for d in derivatives) or any(d < 0 for d in derivatives):
                raise ValueError("Expecting a derivative multi-index with nonnegative indices and not %s" % str(derivatives))
        else:
            derivatives = (0,) * len(self.ufl_operands)
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

    def result_coefficient(self, unpack_reference=True):
        "Returns the coefficient produced by the external operator"
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

    def _analyze_form_arguments(self):
        "Analyze which Argument and Coefficient objects can be found in the base form."
        from ufl.algorithms.analysis import extract_arguments, extract_coefficients
        arguments = tuple(a for arg in self.argument_slots() for a in extract_arguments(arg))
        coefficients = tuple(c for op in self.ufl_operands for c in extract_coefficients(op))
        # Define canonical numbering of arguments and coefficients
        from collections import OrderedDict
        # Need concept of order since we may have arguments with the same number
        # because of `argument_slots(outer_form=True)`:
        #  Example: Let u \in V1 and N \in V2 and F = N(u; v*) * dx, then
        #  `derivative(F, u)` will contain dNdu(u; uhat, v*) with v* = Argument(0, V2)
        #  and uhat = Argument(0, V1) (since F.arguments() = ())
        self._arguments = tuple(sorted(OrderedDict.fromkeys(arguments), key=lambda x: x.number()))
        self._coefficients = tuple(sorted(set(coefficients), key=lambda x: x.count()))

    def _analyze_external_operators(self):
        r"""Analyze which ExternalOperator objects can be found in a given ExternalOperator.
            Example: Let N1, N2 be ExternalOperators and u, w be Coefficients:
                self = N1(u, N2(w); vstar)
                -> self._external_operators = (N2, N1)
        """
        from ufl.algorithms.analysis import extract_external_operators
        extops = (self,) + tuple(e for op in self.ufl_operands for e in extract_external_operators(op))
        self._external_operators = tuple(sorted(set(extops), key=lambda x: x.count()))

    def count(self):
        "Returns the count associated to the coefficient produced by the external operator"
        return self._count

    @property
    def _count(self):
        return self.result_coefficient()._count

    @property
    def ufl_shape(self):
        "Returns the UFL shape of the coefficient.produced by the external operator"
        return self.result_coefficient()._ufl_shape

    def ufl_function_space(self):
        "Returns the ufl function space associated to the external operator"
        return self.result_coefficient()._ufl_function_space

    def grad(self):
        """Returns the symbolic grad of the external operator"""
        # By default, differential rules produce grad(o.result_coefficient()) since
        # the external operator may not be smooth enough for chain rule to hold.
        # Symbolic gradient (grad(ExternalOperator)) depends on the operator considered
        # and its implementation may be needed in some cases (e.g. convolution operator).
        raise NotImplementedError('Symbolic gradient not defined for the external operator considered!')

    def assemble(self, *args, **kwargs):
        """Assemble the external operator"""
        raise NotImplementedError("Symbolic evaluation of %s not available." % self._ufl_class_.__name__)

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, result_coefficient=None, argument_slots=None, add_kwargs={}):
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
                          argument_slots=argument_slots or self.argument_slots(),
                          **add_kwargs)

    def __repr__(self):
        "Default repr string construction for operators."
        r = "ExternalOperator(%s; %s; derivatives=%s, %s)" % (", ".join(repr(op) for op in self.ufl_operands),
                                                              ", ".join(repr(arg) for arg in self.argument_slots()),
                                                              repr(self.derivatives),
                                                              repr(self._count))
        return r

    def __str__(self):
        "Default str string for ExternalOperator operators."
        d = '\N{PARTIAL DIFFERENTIAL}'
        derivatives = self.derivatives
        d_ops = "".join(d + "o" + str(i + 1) for i, di in enumerate(derivatives) for j in range(di))
        e = "e(%s; %s)" % (", ".join(str(op) for op in self.ufl_operands),
                           ", ".join(str(arg) for arg in reversed(self.argument_slots())))
        return d + e + "/" + d_ops if sum(derivatives) > 0 else e

    def _ufl_compute_hash_(self):
        "Default hash of terminals just hash the repr string."
        hashdata = (type(self),
                    # What about Interpolation/ExternalOperator inside operands that
                    # get evaluated and turned into Coefficients ?
                    tuple(type(op) for op in self.ufl_operands),
                    # tuple((type(op), op.ufl_function_space()) for op in self.ufl_operands),
                    self.derivatives,
                    self.ufl_function_space())
        return hash(hashdata)  # hash(repr(self))

    def _ufl_signature_data_(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        coefficient_signature = self.result_coefficient()._ufl_signature_data_(renumbering)
        # TODO: Do we need anything else in the signature?
        return ("ExternalOperator", *coefficient_signature, *self.derivatives)

    def __eq__(self, other):
        if not isinstance(other, ExternalOperator):
            return False
        if self is other:
            return True
        return (type(self) == type(other) and
                # What about Interpolation/ExternalOperator inside operands that
                # get evaluated and turned into Coefficients ?
                all(type(a) == type(b) for a, b in zip(self.ufl_operands, other.ufl_operands)) and
                # all(type(a) == type(b) and a.ufl_function_space() == b.ufl_function_space()
                #    for a, b in zip(self.ufl_operands, other.ufl_operands)) and
                self.derivatives == other.derivatives and
                self.ufl_function_space() == other.ufl_function_space())
        # return (self.count() == other.count() and
        #        self.ufl_function_space() == other.ufl_function_space())
