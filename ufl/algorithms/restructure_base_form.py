"""Restructure BaseForms so that each operand can be assembled independently."""

# Copyright (C) 2026 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from functools import singledispatchmethod

from ufl.action import Action
from ufl.adjoint import Adjoint
from ufl.algebra import Conj, Sum
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.compute_form_data import preprocess_form
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.replace import replace
from ufl.argument import Argument, Coargument
from ufl.core.base_form_operator import (
    BaseFormOperator,
    _number_contracted_argument,
    _split_dual_slot_arguments,
)
from ufl.core.expr import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.form import BaseForm, Form, FormSum


def _rank(o: Expr | BaseForm) -> int:
    """Return the form rank, treating coefficient expressions as rank one."""
    return len(o.arguments()) if isinstance(o, BaseForm) else 1


def _contract_first(form: BaseForm, value: Expr | BaseForm) -> BaseForm:
    """Replace the lowest-numbered argument of ``form``, numbering the others after it."""
    contracted, *others = form.arguments()
    mapping = {a: a.reconstruct(number=a.number() - 1) for a in others}
    return replace(form, {contracted: value, **mapping})


class BaseFormRestructurer(DAGTraverser):
    """Split operator compositions, associate Actions and absorb their operands.

    The operands of an Action are restructured first, and the first rule that matches
    applies. An Action numbers its free arguments as its operands do, whereas Forms and
    operators absorbing an operand number theirs from zero. Rank and function spaces are
    preserved. Here ``N`` is an operator with ``v*`` or ``uhat`` free, ``x`` has rank one,
    ``L`` is a rank one BaseForm, ``A`` and ``B`` have rank two except that the left
    operand of a chain can have a higher rank, ``F`` is a Form and ``w`` is an expression.
    Absorbing comes before distributing, so an operand absorbs a sum whole. A chain of
    rank-two operands is left unchanged, and so is a sum of them, since their assembly
    cost depends on information unavailable here.

    =================================  ===================================================
    Expression                         Result
    =================================  ===================================================
    ``Action(Adjoint(A), x)``          ``Action(x, A)``
    ``Action(x, Adjoint(A))``          ``Action(A, x)``
    ``Action(x, Action(A, B))``        ``Action(Action(x, A), B)``
    ``Action(Action(A, B), x)``        ``Action(A, Action(B, x))``
    ``Action(a * A + b * B, x)``       ``a * Action(A, x) + b * Action(B, x)``
    ``Action(x, a * A + b * B)``       ``conj(a) * Action(x, A) + conj(b) * Action(x, B)``
    ``Action(L, N(u; v_0*, uhat_1))``  ``N(u; L, uhat_0)``
    ``Action(w, N(u; v_1*, uhat_0))``  ``N(u; v_0*, w)``
    ``Action(N(u; v_1*, uhat_0), L)``  ``N(u; L, uhat_0)``
    ``Action(N(u; v_0*, uhat_1), w)``  ``N(u; v_0*, w)``
    ``Action(F(v_0, v_1), w)``         ``F(v_0, w)``
    ``Action(w, F(v_0, v_1))``         ``conj(F(w, v_0))``
    ``Action(a + b, R)``               ``Action(a, R) + Action(b, R)``
    ``Action(L, a + b)``               ``Action(L, a) + Action(L, b)``
    =================================  ===================================================

    The other rules are below, where ``B`` has rank two, ``A`` has rank above one, ``L``
    is a one-form and ``w_i`` is an Argument numbered ``i``. An operator with one
    rank-two primal slot ``B`` is split first.

    ============================  ===============================================
    Expression                    Result
    ============================  ===============================================
    ``N(u; v*, B)``               ``Action(N(u; v*, uhat), B)``
    ``N(u; A, w_0)``              ``Action(N(u; vhat, w_0), A)``
    ``N(u; A, w_1)``              ``Action(A, N(u; vhat, w_1))``
    ``N(u; L(v_1))``              ``N(u; L(v_0))``
    ``Adjoint(N(u; v_0*, w_1))``  ``N(u; v_1*, w_0)``
    ============================  ===============================================
    """

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
        return FormSum(*zip(components, o.weights()))

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> Expr | BaseForm:
        """Split compositions in argument slots into Actions."""
        operands = tuple(self(op) for op in o.ufl_operands)
        argument_slots = tuple(self(slot) for slot in o.argument_slots())
        if operands != o.ufl_operands or argument_slots != o.argument_slots():
            o = o._ufl_expr_reconstruct_(*operands, argument_slots=argument_slots)
        v, *slots = o.argument_slots()
        map_slots = [i for i, slot in enumerate(slots) if _rank(slot) == 2]
        if len(map_slots) == 1:
            # A base form operator is linear in its other argument slots, so for
            # any linear map B: N(u; v*, B) -> Action(N(u; v*, uhat), B), where uhat
            # is numbered after the other arguments of N so that the Action contracts
            # it with the Coargument of B. An Action contracts one argument, so an
            # operator with more than one such slot is not split.
            (i,) = map_slots
            B = slots[i]
            # The Coargument of B is bound inside B, so only the other
            # argument of B moves out of N.
            (Bstar,) = (a for a in B.arguments() if isinstance(a, Coargument))
            (moved,) = (a for a in B.arguments() if a != Bstar)
            others = (a for a in o.arguments() if a != moved)
            number = max((a.number() for a in others), default=-1) + 1
            space = Bstar.ufl_function_space().dual()
            uhat = moved.reconstruct(function_space=space, number=number)
            left = o._ufl_expr_reconstruct_(
                *o.ufl_operands, argument_slots=(v, *slots[:i], uhat, *slots[i + 1 :])
            )
            return self(Action(left, B))

        if isinstance(v, Coargument):
            return o

        if len(v.arguments()) > 1:
            # A base form operator is linear in its dual slot, so N(u; A) splits into
            # Action(N(u; vhat), A) if the primal slots use number 0, and otherwise
            # into Action(A, N(u; vhat)). Each contracts the contracted argument of A.
            A = _number_contracted_argument(o, v, o.arguments())
            slot_arguments = [a for s in slots for a in extract_type(s, Argument, True)]
            contracted, _ = _split_dual_slot_arguments(A, slot_arguments)
            slot_numbers = [a.number() for a in slot_arguments]
            operator_first = 0 in slot_numbers
            space = contracted.ufl_function_space().dual()
            vhat_number = 1 + max(slot_numbers) if operator_first else 0
            vhat = contracted.reconstruct(function_space=space, number=vhat_number)
            Nhat = o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(vhat, *slots))
            return self(Action(Nhat, A) if operator_first else Action(A, Nhat))
        if isinstance(v, Form) and v.arguments()[0].number() != 0:
            # The contracted argument of a one-form is bound in the dual slot, so its
            # number is free. Forms are assembled with arguments numbered from zero.
            (contracted,) = v.arguments()
            dual = replace(v, {contracted: contracted.reconstruct(number=0)})
            return o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(dual, *slots))
        return o

    @process.register(Adjoint)
    @DAGTraverser.postorder
    def _(self, o: Adjoint, form: BaseForm) -> Expr | BaseForm:
        """Swap argument numbers for the Adjoint of a base form operator."""
        if isinstance(form, BaseFormOperator):
            # The adjoint of a base form operator swaps the numbers of its arguments.
            u, v = form.arguments()
            adjoint = replace(
                form, {u: u.reconstruct(number=v.number()), v: v.reconstruct(number=u.number())}
            )
            return self(adjoint)
        return Adjoint(form)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Restructure an Action after processing its operands."""
        left_is_rank_one = _rank(left) == 1
        right_is_rank_one = _rank(right) == 1

        # Contract rank-one operands first, so that every Action in a chain gives
        # another rank-one operand. Reassociate across rank-two operands only: for
        # higher ranks, the argument contracted depends on the position in the chain.
        if isinstance(left, Adjoint) and right_is_rank_one:
            return self(Action(right, left.form()))
        if isinstance(right, Adjoint) and left_is_rank_one:
            return self(Action(right.form(), left))
        if left_is_rank_one and isinstance(right, Action) and _rank(right.left()) == 2:
            return self(Action(Action(left, right.left()), right.right()))
        if (
            right_is_rank_one
            and isinstance(left, Action)
            and _rank(left.left()) > 1
            and _rank(left.right()) == 2
        ):
            return self(Action(left.left(), Action(left.right(), right)))

        # Distribute over a FormSum only to avoid assembling a sum of rank above one.
        if isinstance(left, FormSum) and _rank(left) > 1 and right_is_rank_one:
            return FormSum(
                *((self(Action(c, right)), w) for c, w in zip(left.components(), left.weights()))
            )
        if isinstance(right, FormSum) and _rank(right) > 1 and left_is_rank_one:
            # Action(x, A) contracts x with Adjoint(A), so the weights are conjugated.
            return FormSum(
                *(
                    (self(Action(left, c)), Conj(w))
                    for c, w in zip(right.components(), right.weights())
                )
            )

        # A base form operator is linear in its argument slots. Any BaseForm fills
        # the dual slot, while a primal slot takes an expression.
        if isinstance(right, BaseFormOperator) and left_is_rank_one:
            if isinstance(left, Expr) or right.argument_slots()[0] == right.arguments()[0]:
                return self(_contract_first(right, left))
        if isinstance(left, BaseFormOperator) and right_is_rank_one:
            if isinstance(right, Expr) or left.argument_slots()[0] == left.arguments()[-1]:
                return self(replace(left, {left.arguments()[-1]: right}))

        if isinstance(left, Form) and isinstance(right, Expr) and right_is_rank_one:
            return replace(left, {left.arguments()[-1]: right})
        if isinstance(right, Form) and isinstance(left, Expr) and left_is_rank_one:
            return map_integrands(Conj, _contract_first(right, left))

        # An expression Sum cannot be assembled independently, unlike a FormSum.
        if isinstance(left, Sum):
            return FormSum(*((self(Action(c, right)), 1) for c in left.ufl_operands))
        if isinstance(right, Sum):
            return FormSum(*((self(Action(left, c)), 1) for c in right.ufl_operands))
        return Action(left, right)


def restructure_base_form(form: Expr | BaseForm, complex_mode: bool = False) -> Expr | BaseForm:
    """Expand derivatives and restructure a BaseForm for independent assembly.

    Leave Forms without base form operators unchanged for the form compiler.
    Split compositions in argument slots into Actions, associate rank-one
    contractions first, absorb Action operands into Forms and operator slots,
    and distribute over the remaining expression Sums.

    Args:
        form: The expression or BaseForm to restructure.
        complex_mode: Whether to preserve complex expressions when preprocessing a Form.

    Returns:
        The restructured BaseForm.
    """
    if isinstance(form, Form):
        if not form.base_form_operators():
            return form
        form = preprocess_form(form, complex_mode)
    else:
        form = expand_derivatives(form)
    return BaseFormRestructurer()(form)
