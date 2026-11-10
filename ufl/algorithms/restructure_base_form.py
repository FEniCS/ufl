"""Restructure BaseForms so that each operand can be assembled independently."""

# Copyright (C) 2026 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections import defaultdict
from collections.abc import Iterable
from functools import singledispatchmethod

from ufl.action import Action
from ufl.adjoint import Adjoint
from ufl.algebra import Conj, Sum
from ufl.algorithms.ad import expand_coefficient_derivatives
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.comparison_checker import do_comparison_check
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms.replace import replace
from ufl.algorithms.traversal import iter_expressions
from ufl.argument import Argument, Coargument
from ufl.core.base_form_operator import (
    BaseFormOperator,
    _number_contracted_argument,
    _split_dual_slot_arguments,
)
from ufl.core.expr import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.corealg.traversal import unique_pre_traversal
from ufl.differentiation import BaseFormDerivative, BaseFormOperatorDerivative
from ufl.duals import is_primal
from ufl.form import BaseForm, Form, FormSum
from ufl.formoperators import default_direction_number, derivative


def _rank(o: Expr | BaseForm) -> int:
    """Return the form rank, treating coefficient expressions as rank one."""
    return len(o.arguments()) if isinstance(o, BaseForm) else 1


class BaseFormRestructurer(DAGTraverser):
    """Split operator compositions, associate Actions and compose their operands.

    ``Action(A, x)`` is the product ``A x``. A rank-one left operand enters through the
    antilinear slot, so ``Action(x, A)`` is the row ``x^H A``, whose Adjoint is the column
    ``A^H x``. The operands of an Action are restructured first, and the first rule that
    matches applies. An Action numbers its free arguments as its operands do, whereas a
    Form or operator composed with an operand numbers them from zero. Rank and function
    spaces are preserved.

    Here ``N`` is an operator, ``v*`` its Coargument and ``uhat`` an Argument in a primal
    slot, numbered as subscripted. ``x`` has rank one, ``L`` is a rank one BaseForm and
    ``w`` a rank one expression. ``A`` and ``B`` have rank two, except that the left
    operand of a chain can have a higher rank. ``F`` is a Form, ``c`` and ``d`` are
    weights and ``y + z`` is an expression Sum. Composing comes before distributing,
    so an operand composes with a sum whole. A product or sum of rank-two operands stays as it
    is, since its assembly cost depends on information unavailable here.

    =================================  ===================================================
    Expression                         Result
    =================================  ===================================================
    ``Action(Adjoint(A), x)``          ``Adjoint(Action(x, A))``
    ``Action(x, Adjoint(A))``          ``Action(A, x)``
    ``Action(x, Action(A, B))``        ``Action(Action(x, A), B)``
    ``Action(Action(A, B), x)``        ``Action(A, Action(B, x))``
    ``Action(c * A + d * B, x)``       ``c * Action(A, x) + d * Action(B, x)``
    ``Action(x, c * A + d * B)``       ``c * Action(x, A) + d * Action(x, B)``
    ``Action(L, N(u; v_0*, uhat_1))``  ``N(u; L, uhat_0)``
    ``Action(w, N(u; v_1*, uhat_0))``  ``N(u; v_0*, w)``
    ``Action(N(u; v_1*, uhat_0), L)``  ``N(u; L, uhat_0)``
    ``Action(N(u; v_0*, uhat_1), w)``  ``N(u; v_0*, w)``
    ``Action(F(v_0, v_1), w)``         ``F(v_0, w)``
    ``Action(w, F(v_0, v_1))``         ``conj(F(w, v_0))``
    ``Action(F(v_0, v_1), N)``         ``F(v_0, N)``
    ``Action(N, F(v_0, v_1))``         ``conj(F(N, v_1))``
    ``Action(y + z, A)``               ``Action(y, A) + Action(z, A)``
    ``Action(A, y + z)``               ``Action(A, y) + Action(A, z)``
    =================================  ===================================================

    The rules for operators follow, where ``B`` has rank two, ``A`` is a Form or has rank
    above one and ``w_i`` is an Argument numbered ``i``. An operator with one rank-two
    primal slot ``B`` is split first. A Form and an operator compose only if the operator
    has rank at most ``max_composed_rank``. Otherwise a rank-one Form stays in the dual
    slot of the operator, and the Action of a Form of higher rank is kept.

    ============================  ===============================================
    Expression                    Result
    ============================  ===============================================
    ``N(u; v*, B)``               ``Action(N(u; v*, uhat), B)``
    ``N(u; A, w_0)``              ``Action(N(u; vhat, w_0), A)``
    ``N(u; A, w_1)``              ``Action(A, N(u; vhat, w_1))``
    ``Adjoint(N(u; v_0*, w_1))``  ``N(u; v_1*, w_0)``
    ``Adjoint(F(x, v_0))``        ``F(x, v_0)``
    ============================  ===============================================
    """

    def __init__(self, max_composed_rank: int = 1, **kwargs) -> None:
        """Initialise.

        Args:
            max_composed_rank: The highest rank of an operator that composes with a Form.
            kwargs: Passed to the DAGTraverser.
        """
        super().__init__(**kwargs)
        self.max_composed_rank = max_composed_rank

    def composable(self, operator: BaseFormOperator) -> bool:
        """Return whether the operator may compose with a Form."""
        return _rank(operator) <= self.max_composed_rank

    def compose(self, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Return ``left(right)`` if ``left`` is a Form, else ``right(left)``.

        The Form takes the other operand into its last or first argument, as
        ``Action(left, right)`` contracts. A function ``left`` leaves the linear slot of
        ``right`` free, so it gives ``conj(right(left))`` to keep the test argument
        antilinear. Free arguments are renumbered from zero, and each bound Coargument
        takes the lowest number its operator leaves free. The Action of an operator
        that is not composable is kept.
        """
        for operand in (left, right):
            if isinstance(operand, BaseFormOperator) and not self.composable(operand):
                return Action(left, right)

        left_free = left.arguments()[:-1] if isinstance(left, BaseForm) else ()
        right_free = right.arguments()[1:] if isinstance(right, BaseForm) else ()
        mapping = {a: a.reconstruct(number=i) for i, a in enumerate((*left_free, *right_free))}
        used: defaultdict[Coargument, set[int]] = defaultdict(set)
        for operand in (left, right):
            for e in iter_expressions(operand):
                for N in unique_pre_traversal(e):
                    if not isinstance(N, BaseFormOperator):
                        continue
                    vstar, *slots = N.argument_slots()
                    if not isinstance(vstar, Coargument):
                        continue
                    for slot in (*N.ufl_operands, *slots):
                        arguments = extract_type(slot, Argument, True)
                        used[vstar].update(mapping.get(a, a).number() for a in arguments)
        for vstar, numbers in used.items():
            mapping[vstar] = vstar.reconstruct(number=min(set(range(len(numbers) + 1)) - numbers))

        if isinstance(left, Form):
            return replace(left, {**mapping, left.arguments()[-1]: replace(right, mapping)})
        if isinstance(right, Form):
            composed = replace(right, {**mapping, right.arguments()[0]: replace(left, mapping)})
            if not isinstance(left, BaseForm):
                return map_integrands(Conj, composed)
            if isinstance(left, BaseFormOperator) and _rank(left) == 1:
                if is_primal(left.ufl_function_space()):
                    return map_integrands(Conj, composed)
            return composed
        raise ValueError("Expecting a Form operand.")

    def column_sum(self, rows: Iterable[BaseForm], weights: Iterable) -> BaseForm:
        """Return the column conj(c) A^H x + conj(d) B^H x of the sum of rows c x^H A + d x^H B.

        Composing a row with its left operand gives the column, so such a row stays as it is.
        """
        columns = [r if r.arguments()[0].number() == 0 else self(Adjoint(r)) for r in rows]
        return FormSum(*zip(columns, map(Conj, weights)))

    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``."""
        return super().process(o)

    @process.register(BaseForm)
    def _(self, o: BaseForm) -> BaseForm:
        """Leave Forms and terminal BaseForms unchanged."""
        return o

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        """Traverse expressions in operator operands and argument slots."""
        return self.reuse_if_untouched(o)

    @process.register(FormSum)
    @DAGTraverser.postorder
    def _(self, o: FormSum, *components: BaseForm) -> BaseForm:
        """Traverse the components of a FormSum."""
        is_row = _rank(o) == 1 and o.arguments()[0].number() > 0
        if is_row and any(c.arguments()[0].number() == 0 for c in components):
            return self.column_sum(components, o.weights())
        return FormSum(*zip(components, o.weights()))

    @process.register(BaseFormDerivative)
    @process.register(BaseFormOperatorDerivative)
    def _(self, o: BaseFormDerivative) -> BaseForm:
        """Restructure the operand of a derivative and renumber its direction to match."""
        operand, coefficients, directions, coefficient_derivatives = o.ufl_operands
        restructured = self(operand)
        if restructured == operand:
            return o
        # A direction numbered as a default one stays right after the free arguments.
        first = default_direction_number(operand)
        shift = default_direction_number(restructured) - first
        mapping = {
            a: a.reconstruct(number=a.number() + shift)
            for a in extract_type(directions, Argument)
            if a.number() >= first
        }
        cd = coefficient_derivatives.ufl_operands
        return derivative(
            restructured,
            coefficients.ufl_operands,
            tuple(replace(v, mapping) for v in directions.ufl_operands),
            dict(zip(cd[::2], cd[1::2])),
        )

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> Expr | BaseForm:
        """Split compositions in argument slots into Actions."""
        operands = tuple(self(op) for op in o.ufl_operands)
        argument_slots = tuple(self(slot) for slot in o.argument_slots())
        if operands != o.ufl_operands or argument_slots != o.argument_slots():
            o = o._ufl_expr_reconstruct_(*operands, argument_slots=argument_slots)
        vstar, *slots = o.argument_slots()
        rank_two_slots = [i for i, slot in enumerate(slots) if _rank(slot) == 2]
        if len(rank_two_slots) == 1:
            # A base form operator is linear in its other argument slots, so for
            # any linear map B: N(u; v*, B) -> Action(N(u; v*, uhat), B), where uhat
            # is numbered after the other arguments of N so that the Action contracts
            # it with the Coargument of B. An Action contracts one argument, so an
            # operator with more than one such slot is not split.
            (i,) = rank_two_slots
            B = slots[i]
            # The Coargument of B is contracted by the Action, so
            # only the other argument of B stays free.
            (vstar_B,) = (a for a in B.arguments() if isinstance(a, Coargument))
            (uhat_B,) = (a for a in B.arguments() if a != vstar_B)
            others = (a for a in o.arguments() if a != uhat_B)
            uhat_number = max((a.number() for a in others), default=-1) + 1
            space = vstar_B.ufl_function_space().dual()
            uhat = uhat_B.reconstruct(function_space=space, number=uhat_number)
            Nhat = o._ufl_expr_reconstruct_(
                *o.ufl_operands, argument_slots=(vstar, *slots[:i], uhat, *slots[i + 1 :])
            )
            return self(Action(Nhat, B))

        if isinstance(vstar, Coargument):
            return o

        if isinstance(vstar, Form) or len(vstar.arguments()) > 1:
            # A base form operator is linear in its dual slot, so N(u; A) splits into
            # Action(N(u; vhat), A) if the primal slots use number 0, and otherwise
            # into Action(A, N(u; vhat)). Each contracts the contracted argument of A.
            A = _number_contracted_argument(o, vstar, o.arguments())
            slot_arguments = [a for s in slots for a in extract_type(s, Argument, True)]
            contracted, _ = _split_dual_slot_arguments(A, slot_arguments)
            slot_numbers = [a.number() for a in slot_arguments]
            operator_first = 0 in slot_numbers
            space = contracted.ufl_function_space().dual()
            vhat_number = 1 + max(slot_numbers) if operator_first else 0
            vhat = contracted.reconstruct(function_space=space, number=vhat_number)
            Nhat = o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(vhat, *slots))
            if isinstance(A, Form):
                if _rank(A) == 1 and not self.composable(Nhat):
                    # A rank-one Form stays in the dual slot of an operator it does not
                    # compose with.
                    return o
                return self.compose(Nhat, A) if operator_first else self.compose(A, Nhat)
            return self(Action(Nhat, A) if operator_first else Action(A, Nhat))
        return o

    @process.register(Adjoint)
    @DAGTraverser.postorder
    def _(self, o: Adjoint, form: BaseForm) -> Expr | BaseForm:
        """Swap argument numbers for the Adjoint of a base form operator."""
        is_row = _rank(form) == 1 and form.arguments()[0].number() > 0
        if is_row:
            if isinstance(form, FormSum):
                return self.column_sum(form.components(), form.weights())
            return Adjoint(form)
        if _rank(form) == 1:
            # form is already a column vector.
            return form
        if isinstance(form, BaseFormOperator):
            # Adjoint(N(u; v_0*, w_1)) -> N(u; v_1*, w_0)
            first, last = form.arguments()
            swapped = replace(
                form,
                {
                    first: first.reconstruct(number=last.number()),
                    last: last.reconstruct(number=first.number()),
                },
            )
            return self(swapped)
        return Adjoint(form)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Restructure an Action after processing its operands."""
        left_is_rank_one = _rank(left) == 1
        right_is_rank_one = _rank(right) == 1

        # Contract rank-one operands first, so that every Action in a chain gives another
        # rank-one operand. Associate across rank-two operands only: for higher ranks,
        # the argument that an Action contracts depends on its position in the chain.
        if isinstance(left, Adjoint) and _rank(left.form()) == 2 and right_is_rank_one:
            # A^H x = (x^H A)^H
            return self(Adjoint(Action(right, left.form())))
        if isinstance(right, Adjoint) and _rank(right.form()) == 2 and left_is_rank_one:
            # (A^H)^H x = A x
            return self(Action(right.form(), left))
        if left_is_rank_one and isinstance(right, Action) and _rank(right.left()) == 2:
            # x^H (A B) = (x^H A) B
            return self(Action(Action(left, right.left()), right.right()))
        if (
            right_is_rank_one
            and isinstance(left, Action)
            and _rank(left.left()) > 1
            and _rank(left.right()) == 2
        ):
            # (A B) x = A (B x)
            return self(Action(left.left(), Action(left.right(), right)))

        # Distribute over a FormSum only to avoid assembling a sum of rank above one.
        if isinstance(left, FormSum) and _rank(left) > 1 and right_is_rank_one:
            # (c A + d B) x = c A x + d B x
            return FormSum(
                *((self(Action(A, right)), c) for A, c in zip(left.components(), left.weights()))
            )
        if isinstance(right, FormSum) and _rank(right) > 1 and left_is_rank_one:
            # x^H (c A + d B) = c x^H A + d x^H B
            rows = [Action(left, A) for A in right.components()]
            return self(FormSum(*zip(rows, right.weights())))

        # An operator is linear in its argument slots. The dual slot takes any
        # rank-one BaseForm, while a primal slot takes only an expression.
        if isinstance(right, BaseFormOperator) and left_is_rank_one:
            contracts_dual_slot = right.argument_slots()[0] == right.arguments()[0]
            if isinstance(left, Expr) or contracts_dual_slot:
                contracted, *free = right.arguments()
                mapping = {a: a.reconstruct(number=a.number() - 1) for a in free}
                return self(replace(right, {contracted: left, **mapping}))
        if isinstance(left, BaseFormOperator) and right_is_rank_one:
            contracts_dual_slot = left.argument_slots()[0] == left.arguments()[-1]
            if isinstance(right, Expr) or contracts_dual_slot:
                return self(replace(left, {left.arguments()[-1]: right}))

        # A Form composes with an expression, or with an operator through its dual slot.
        if isinstance(left, Form) and isinstance(right, BaseFormOperator):
            if right.argument_slots()[0] == right.arguments()[0]:
                return self.compose(left, right)
        elif isinstance(left, Form) and isinstance(right, Expr):
            return self.compose(left, right)
        if isinstance(right, Form) and isinstance(left, BaseFormOperator):
            if left.argument_slots()[0] == left.arguments()[-1]:
                return self.compose(left, right)
        elif isinstance(right, Form) and isinstance(left, Expr):
            return self.compose(left, right)

        # An expression Sum cannot be assembled independently, unlike a FormSum.
        if isinstance(left, Sum):
            return FormSum(*((self(Action(w, right)), 1) for w in left.ufl_operands))
        if isinstance(right, Sum):
            return FormSum(*((self(Action(left, w)), 1) for w in right.ufl_operands))
        return Action(left, right)


def restructure_base_form(
    form: Expr | BaseForm, complex_mode: bool = False, max_composed_rank: int = 1
) -> Expr | BaseForm:
    """Expand coefficient derivatives and restructure a BaseForm for independent assembly.

    Leave Forms without base form operators unchanged for the form compiler.
    Preserve spatial derivatives, split compositions in argument slots into
    Actions, associate rank-one contractions first, compose Forms and operators
    with Action operands, and distribute over the remaining expression Sums.

    Args:
        form: The expression or BaseForm to restructure.
        complex_mode: Whether to preserve complex expressions when preprocessing a Form.
        max_composed_rank: The highest rank of an operator that composes with a Form.
            A rank-one Form stays in the dual slot of an operator of higher rank.

    Returns:
        The restructured BaseForm.
    """
    restructurer = BaseFormRestructurer(max_composed_rank)
    if isinstance(form, Form):
        if not form.base_form_operators():
            return form
        if complex_mode:
            form = do_comparison_check(form)
        form = expand_coefficient_derivatives(form)
        if not complex_mode:
            form = remove_complex_nodes(form)
        return restructurer(form)
    # Restructure before expanding derivatives too, so that each derivative
    # numbers its direction after the free arguments of its restructured operand.
    return restructurer(expand_coefficient_derivatives(restructurer(form)))
