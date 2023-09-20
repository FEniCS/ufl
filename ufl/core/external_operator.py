"""External operator.

This module defines the ``ExternalOperator`` class, which symbolically represents operators that are not
straightforwardly expressible in UFL. Subclasses of ``ExternalOperator`` must define
how this operator should be evaluated as well as its derivatives from a given set of operands.
"""
# Copyright (C) 2019 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2023

from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.ufl_type import ufl_type


@ufl_type(num_ops="varying", is_differential=True)
class ExternalOperator(BaseFormOperator):
    """External operator."""

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=()):
        """Initialise.

        Args:
            operands: operands on which acts the ExternalOperator.
            function_space: the FunctionSpace, or MixedFunctionSpace on which to build this Function.
            derivatives: tuple specifiying the derivative multiindex.
            argument_slots: tuple composed containing expressions with ufl.Argument or ufl.Coefficient objects.
        """
        # -- Derivatives -- #
        if derivatives is not None:
            if not isinstance(derivatives, tuple):
                raise TypeError(f"Expecting a tuple for derivatives and not {derivatives}")
            if not len(derivatives) == len(operands):
                raise ValueError(f"Expecting a size of {len(operands)} for {derivatives}")
            if not all(isinstance(d, int) for d in derivatives) or any(d < 0 for d in derivatives):
                raise ValueError(
                    f"Expecting a derivative multi-index with nonnegative indices and not {str(derivatives)}")
        else:
            derivatives = (0,) * len(operands)

        BaseFormOperator.__init__(self, *operands,
                                  function_space=function_space,
                                  derivatives=derivatives,
                                  argument_slots=argument_slots)

    def ufl_element(self):
        """Shortcut to get the finite element of the function space of the external operator."""
        # Useful when applying split on an ExternalOperator
        return self.arguments()[0].ufl_element()

    def grad(self):
        """Returns the symbolic grad of the external operator."""
        # By default, differential rules produce `grad(assembled_o)` `where assembled_o`
        # is the `Coefficient` resulting from assembling the external operator since
        # the external operator may not be smooth enough for chain rule to hold.
        # Symbolic gradient (`grad(ExternalOperator)`) depends on the operator considered
        # and its implementation may be needed in some cases (e.g. convolution operator).
        raise NotImplementedError('Symbolic gradient not defined for the external operator considered!')

    def assemble(self, *args, **kwargs):
        """Assemble the external operator."""
        raise NotImplementedError(f"Symbolic evaluation of {self._ufl_class_.__name__} not available.")

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None,
                               argument_slots=None, add_kwargs={}):
        """Return a new object of the same type with new operands."""
        return type(self)(*operands, function_space=function_space or self.ufl_function_space(),
                          derivatives=derivatives or self.derivatives,
                          argument_slots=argument_slots or self.argument_slots(),
                          **add_kwargs)

    def __str__(self):
        """Default str string for ExternalOperator operators."""
        d = '\N{PARTIAL DIFFERENTIAL}'
        derivatives = self.derivatives
        d_ops = "".join(d + "o" + str(i + 1) for i, di in enumerate(derivatives) for j in range(di))
        e = "e("
        e += ", ".join(str(op) for op in self.ufl_operands)
        e += "; "
        e += ", ".join(str(arg) for arg in reversed(self.argument_slots()))
        e += ")"
        return d + e + "/" + d_ops if sum(derivatives) > 0 else e

    def __eq__(self, other):
        """Check for equality."""
        if self is other:
            return True
        return (type(self) is type(other) and
                all(a == b for a, b in zip(self.ufl_operands, other.ufl_operands)) and
                all(a == b for a, b in zip(self._argument_slots, other._argument_slots)) and
                self.derivatives == other.derivatives and
                self.ufl_function_space() == other.ufl_function_space())
