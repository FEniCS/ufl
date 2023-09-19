"""Base form operator.

This module defines the BaseFormOperator class, which is the base class for objects that can be seen as forms
and as operators such as ExternalOperator or Interpolate.
"""

# Copyright (C) 2019 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022

from collections import OrderedDict

from ufl.argument import Argument, Coargument
from ufl.core.operator import Operator
from ufl.form import BaseForm
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import as_ufl
from ufl.functionspace import AbstractFunctionSpace
from ufl.utils.counted import Counted


@ufl_type(num_ops="varying", is_differential=True)
class BaseFormOperator(Operator, BaseForm, Counted):
    """Base form operator."""

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=()):
        """Initialise.

        Args:
            operands: operands on which acts the operator.
            function_space: the FunctionSpace or MixedFunctionSpace on which to build this Function.
            derivatives: tuple specifiying the derivative multiindex.
            argument_slots: tuple composed containing expressions with ufl.Argument or ufl.Coefficient objects.
        """
        BaseForm.__init__(self)
        ufl_operands = tuple(map(as_ufl, operands))
        argument_slots = tuple(map(as_ufl, argument_slots))
        Operator.__init__(self, ufl_operands)
        Counted.__init__(self, counted_class=BaseFormOperator)

        # -- Function space -- #
        if not isinstance(function_space, AbstractFunctionSpace):
            raise ValueError("Expecting a FunctionSpace or FiniteElement.")

        # -- Derivatives -- #
        # Some BaseFormOperator does have derivatives (e.g. ExternalOperator)
        # while other don't since they are fully determined by their
        # argument slots (e.g. Interpolate)
        self.derivatives = derivatives

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

    def argument_slots(self, outer_form=False):
        """Returns a tuple of expressions containing argument and coefficient based expressions.

        We get an argument uhat when we take the Gateaux derivative in the direction uhat:
        d/du N(u; v*) = dNdu(u; uhat, v*) where uhat is a ufl.Argument and v* a ufl.Coargument
        Applying the action replace the last argument by coefficient:
        action(dNdu(u; uhat, v*), w) = dNdu(u; w, v*) where du is a ufl.Coefficient.
        """
        from ufl.algorithms.analysis import extract_arguments
        if not outer_form:
            return self._argument_slots
        # Takes into account argument contraction when a base form operator is in an outer form:
        # For example:
        #   F = N(u; v*) * v * dx can be seen as Action(v1 * v * dx, N(u; v*))
        #   => F.arguments() should return (v,)!
        return tuple(a for a in self._argument_slots[1:] if len(extract_arguments(a)) != 0)

    def coefficients(self):
        """Return all BaseCoefficient objects found in base form operator."""
        if self._coefficients is None:
            self._analyze_form_arguments()
        return self._coefficients

    def _analyze_form_arguments(self):
        """Analyze which Argument and Coefficient objects can be found in the base form."""
        from ufl.algorithms.analysis import extract_arguments, extract_coefficients, extract_type
        dual_arg, *arguments = self.argument_slots()
        # When coarguments are treated as BaseForms, they have two arguments (one primal and one dual)
        # as they map from V* to V* => V* x V -> R. However, when they are treated as mere "arguments",
        # the primal space argument is discarded and we only have the dual space argument (Coargument).
        # This is the exact same situation than BaseFormOperator's arguments which are different depending on
        # whether the BaseFormOperator is used in an outer form or not.
        arguments = (tuple(extract_type(dual_arg, Coargument))
                     + tuple(a for arg in arguments for a in extract_arguments(arg)))
        coefficients = tuple(c for op in self.ufl_operands for c in extract_coefficients(op))
        # Define canonical numbering of arguments and coefficients
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
        """Return the count associated to the base form operator."""
        return self._count

    @property
    def ufl_shape(self):
        """Return the UFL shape of the coefficient.produced by the operator."""
        return self.arguments()[0]._ufl_shape

    def ufl_function_space(self):
        """Return the function space associated to the operator.

        I.e. return the dual of the base form operator's Coargument.
        """
        return self.arguments()[0]._ufl_function_space.dual()

    def _ufl_expr_reconstruct_(self, *operands, function_space=None, derivatives=None, argument_slots=None):
        """Return a new object of the same type with new operands."""
        return type(self)(*operands, function_space=function_space or self.ufl_function_space(),
                          derivatives=derivatives or self.derivatives,
                          argument_slots=argument_slots or self.argument_slots())

    def __repr__(self):
        """Default repr string construction for base form operators."""
        r = f"{type(self).__name__}("
        r += ", ".join(repr(op) for op in self.ufl_operands)
        r += "; {self.ufl_function_space()!r}; "
        r += ", ".join(repr(arg) for arg in self.argument_slots())
        r += f"; derivatives={self.derivatives!r})"
        return r

    def __hash__(self):
        """Hash code for use in dicts."""
        hashdata = (type(self),
                    tuple(hash(op) for op in self.ufl_operands),
                    tuple(hash(arg) for arg in self._argument_slots),
                    self.derivatives,
                    hash(self.ufl_function_space()))
        return hash(hashdata)

    def __eq__(self, other):
        """Check for equality."""
        raise NotImplementedError()
