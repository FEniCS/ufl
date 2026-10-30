"""Base form operator.

This module defines the BaseFormOperator class, which is the base class
for objects that can be seen as forms and as operators such as
ExternalOperator or Interpolate.
"""

# Copyright (C) 2019 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022

from collections import OrderedDict
from itertools import chain, count
from numbers import Number

from ufl.argument import Argument, Coargument
from ufl.coefficient import Cofunction
from ufl.constantvalue import as_ufl
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.duals import is_primal
from ufl.form import BaseForm
from ufl.functionspace import AbstractFunctionSpace
from ufl.utils.counted import Counted

__all__ = ["BaseFormOperator"]


def _is_nonterminal_base_form_dual_slot(dual_arg):
    """Return whether a dual slot is a non-terminal BaseForm."""
    return isinstance(dual_arg, BaseForm) and not isinstance(dual_arg, (Coargument, Cofunction))


def _split_dual_slot_arguments(dual_arg):
    """Return the contracted argument and the free arguments of a dual slot.

    A Coargument or a Cofunction contracts its primal argument, which comes
    first. Any other BaseForm contracts its highest-numbered primal argument,
    as the left operand of an Action does. Its Coarguments are outputs, so
    they stay free.
    """
    arguments = dual_arg.arguments()
    if not _is_nonterminal_base_form_dual_slot(dual_arg):
        return (arguments[0], arguments[1:]) if arguments else (None, ())
    primal_arguments = [a for a in arguments if not isinstance(a, Coargument)]
    if not primal_arguments:
        return None, arguments
    contracted = primal_arguments[-1]
    return contracted, tuple(a for a in arguments if a != contracted)


def _renumber_dual_slot_arguments(dual_arg, primal_args):
    """Return the renumbering of the arguments in a non-terminal BaseForm dual slot.

    The free arguments take, in order, the lowest numbers that the primal
    arguments do not use. The contracted argument takes the lowest number
    above them that does not make it equal to an argument of the operator, so
    it stays the highest-numbered primal argument of the slot.
    """
    contracted, free = _split_dual_slot_arguments(dual_arg)
    primal_numbers = {argument.number() for argument in primal_args}
    numbers = (number for number in count() if number not in primal_numbers)
    renumbered = tuple(argument.reconstruct(number=n) for argument, n in zip(free, numbers))
    exposed = {*renumbered, *primal_args}
    lowest = max((a.number() + 1 for a in renumbered if not isinstance(a, Coargument)), default=0)
    number = next(n for n in count(lowest) if contracted.reconstruct(number=n) not in exposed)
    renumbering = dict(zip(free, renumbered))
    renumbering[contracted] = contracted.reconstruct(number=number)
    return {old: new for old, new in renumbering.items() if old != new}


@ufl_type(num_ops="varying", is_differential=True)
class BaseFormOperator(Operator, BaseForm, Counted):
    """Base form operator."""

    ufl_operands: tuple[Expr, ...]

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, *operands, function_space, derivatives=None, argument_slots=()):
        """Initialise.

        Args:
            operands:
                The expressions that the operator acts on. The operator can be
                nonlinear in them.
            function_space:
                The FunctionSpace or MixedFunctionSpace of the operator output.
            derivatives:
                The derivative multi-index: the number of times that the
                operator is differentiated with respect to each operand.
            argument_slots:
                The slots in which the operator is linear, given as
                ``(dual_slot, *primal_slots)``. The dual slot is paired with the
                operator output, and it can be one of these:

                - A Coargument on the dual of ``function_space``. The output
                  stays free, so the operator is a function in
                  ``function_space``.
                - A Cofunction on the dual of ``function_space``, which
                  contracts the output to the number ``<c, N(u)>``.
                - A Form whose highest-numbered Argument is on
                  ``function_space``. The output takes the place of that
                  Argument, which gives ``F(N(u))``, as in
                  ``Action(F, N(u))``. The other Arguments of the form stay
                  free. They are renumbered to the lowest numbers that the
                  primal slots do not use, so replacing an Argument in a
                  primal slot can change their numbers.

                Each primal slot holds a direction of differentiation: an
                expression of an Argument, or of a Coefficient after an action.
                If ``argument_slots`` is empty, the dual slot is a new Coargument
                with number 0 on the dual of ``function_space``.
        """
        # Avoid cyclic import
        from ufl.algorithms.analysis import extract_type
        from ufl.algorithms.replace import replace

        BaseForm.__init__(self)
        ufl_operands = tuple(map(as_ufl, operands))
        argument_slots = tuple(map(as_ufl, argument_slots))
        Operator.__init__(self, ufl_operands)
        Counted.__init__(self, counted_class=BaseFormOperator)

        # -- Function space -- #
        if not isinstance(function_space, AbstractFunctionSpace):
            raise ValueError("Expecting a FunctionSpace or FiniteElement.")
        self._function_space = function_space

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

        dual_slot = argument_slots[0]
        if not isinstance(dual_slot, BaseForm):
            raise ValueError("The dual slot must be a BaseForm.")

        if _is_nonterminal_base_form_dual_slot(dual_slot) and dual_slot.arguments():
            primal_args = {
                a
                for slot in argument_slots[1:]
                for a in extract_type(slot, Argument, base_form_op_as_expr=True)
            }
            renumbering = _renumber_dual_slot_arguments(dual_slot, primal_args)
            if renumbering:
                # Keep the arguments exposed by this operator in its stored slot.
                dual_slot = replace(dual_slot, renumbering)
                argument_slots = (dual_slot, *argument_slots[1:])
        self._argument_slots = argument_slots

        contracted, _ = _split_dual_slot_arguments(dual_slot)
        if contracted is not None and contracted.ufl_function_space() != function_space:
            raise ValueError("The contracted argument in the dual slot must be on function_space.")

        # Internal variables for caching coefficient and domain data
        self._coefficients = None
        self._domains = None

    # BaseFormOperators don't have free indices.
    ufl_free_indices = ()
    ufl_index_dimensions = ()

    def argument_slots(self):
        """Returns a tuple of expressions containing argument and coefficient based expressions.

        We get an argument uhat when we take the Gateaux derivative in
        the direction uhat: d/du N(u; v*) = dNdu(u; uhat, v*) where uhat
        is a ufl.Argument and v* a ufl.Coargument Applying the action
        replace the last argument by coefficient: action(dNdu(u; uhat,
        v*), w) = dNdu(u; w, v*) where du is a ufl.Coefficient.
        """
        return self._argument_slots

    def _analyze_domains(self) -> None:
        """Analyze domains in the operands and the argument slots."""
        from ufl.domain import extract_domains, join_domains, sort_domains

        def extract(expression):
            if isinstance(expression, BaseForm):
                return expression.ufl_domains()
            return extract_domains(expression)

        expressions = (*self.ufl_operands, *self.argument_slots())
        self._domains = sort_domains(
            join_domains(list(chain.from_iterable(extract(e) for e in expressions)))
        )

    def ufl_domains(self):
        """Return all domains found in the operands and the argument slots."""
        if self._domains is None:
            self._analyze_domains()
        return self._domains

    def coefficients(self):
        """Return all BaseCoefficient objects found in base form operator."""
        if self._coefficients is None:
            self._analyze_form_arguments()
        return self._coefficients

    def _analyze_form_arguments(self):
        """Analyze which Argument and Coefficient objects can be found in the base form."""
        from ufl.algorithms.analysis import extract_coefficients, extract_type

        dual_arg, *arguments = self.argument_slots()
        primal_args = tuple(
            a for arg in arguments for a in extract_type(arg, Argument, base_form_op_as_expr=True)
        )
        # The stored dual slot is already renumbered.
        _, dual_args = _split_dual_slot_arguments(dual_arg)
        arguments = dual_args + primal_args
        coefficients = tuple(c for op in self.ufl_operands for c in extract_coefficients(op))
        # Define canonical numbering of arguments and coefficients
        # 1) Need concept of order since we may have arguments with the same number
        #    because of form composition:
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
        """Return the UFL shape of the coefficient produced by the operator."""
        if len(self.arguments()) == 0:
            return ()
        arg, *_ = self.arguments()
        return arg._ufl_shape

    def ufl_function_space(self):
        """Return the function space associated to the operator.

        I.e. return the dual of the base form operator's Coargument space.
        """
        if len(self.arguments()) == 0:
            return None
        arg, *_ = self.arguments()
        return arg.ufl_function_space().dual()

    def _ufl_expr_reconstruct_(
        self, *operands, function_space=None, derivatives=None, argument_slots=None
    ):
        """Return a new object of the same type with new operands."""
        return type(self)(
            *operands,
            function_space=function_space or self._function_space,
            derivatives=derivatives or self.derivatives,
            argument_slots=argument_slots or self.argument_slots(),
        )

    def __repr__(self):
        """Default repr string construction for base form operators."""
        r = f"{type(self).__name__}("
        r += ", ".join(repr(op) for op in self.ufl_operands)
        r += "; {self.ufl_function_space()!r}; "
        r += ", ".join(repr(arg) for arg in self.argument_slots())
        r += f"; derivatives={self.derivatives!r})"
        return r

    def _ufl_compute_hash_(self):
        """Hash code for use in dicts."""
        hashdata = (
            self._ufl_typecode_,
            tuple(hash(op) for op in self.ufl_operands),
            tuple(hash(arg) for arg in self._argument_slots),
            self.derivatives,
            hash(self._function_space),
        )
        return hash(hashdata)

    def __eq__(self, other):
        """Check for equality."""
        if isinstance(other, Number) and other == 0:
            return self.empty()
        raise NotImplementedError()

    @property
    def _parent_type(self):
        """Is this a primal or dual expression?"""
        return Operator if is_primal(self.ufl_function_space()) else BaseForm

    def __add__(self, other):
        """Add."""
        return self._parent_type.__add__(self, other)

    def __radd__(self, other):
        """Add."""
        return self._parent_type.__radd__(self, other)

    def __mul__(self, other):
        """Multiply."""
        return self._parent_type.__mul__(self, other)

    def __rmul__(self, other):
        """Multiply."""
        return self._parent_type.__rmul__(self, other)
