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

from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.ufl_type import ufl_type


@ufl_type(num_ops="varying", is_differential=True)
class ExternalOperator(BaseFormOperator):

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

        # -- Derivatives -- #
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                raise TypeError("Expecting a tuple for derivatives and not %s" % derivatives)
            if not len(derivatives) == len(operands):
                raise ValueError("Expecting a size of %s for %s" % (len(operands), derivatives))
            if not all(isinstance(d, int) for d in derivatives) or any(d < 0 for d in derivatives):
                raise ValueError("Expecting a derivative multi-index with nonnegative indices and not %s" % str(derivatives))
        else:
            derivatives = (0,) * len(operands)

        BaseFormOperator.__init__(self, *operands,
                                  function_space=function_space,
                                  derivatives=derivatives,
                                  result_coefficient=result_coefficient,
                                  argument_slots=argument_slots)

    def _analyze_external_operators(self):
        r"""Analyze which ExternalOperator objects can be found in a given ExternalOperator.
            Example: Let N1, N2 be ExternalOperators and u, w be Coefficients:
                self = N1(u, N2(w); vstar)
                -> self._external_operators = (N2, N1)
        """
        from ufl.algorithms.analysis import extract_external_operators
        extops = (self,) + tuple(e for op in self.ufl_operands for e in extract_external_operators(op))
        self._external_operators = tuple(sorted(set(extops), key=lambda x: x.count()))

    """
    def ufl_element(self):
        "Shortcut to get the finite element of the function space of the external operator"
        return self.result_coefficient().ufl_element()
    """

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

    def __str__(self):
        "Default str string for ExternalOperator operators."
        d = '\N{PARTIAL DIFFERENTIAL}'
        derivatives = self.derivatives
        d_ops = "".join(d + "o" + str(i + 1) for i, di in enumerate(derivatives) for j in range(di))
        e = "e(%s; %s)" % (", ".join(str(op) for op in self.ufl_operands),
                           ", ".join(str(arg) for arg in reversed(self.argument_slots())))
        return d + e + "/" + d_ops if sum(derivatives) > 0 else e

    def __eq__(self, other):
        if not isinstance(other, ExternalOperator):
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
