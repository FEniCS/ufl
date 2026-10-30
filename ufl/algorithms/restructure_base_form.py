"""Restructure BaseForms so that each operand can be assembled independently."""

# Copyright (C) 2026 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections.abc import Callable
from functools import singledispatchmethod
from numbers import Complex

from ufl.action import Action
from ufl.adjoint import Adjoint
from ufl.algebra import Conj, Sum
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.compute_form_data import preprocess_form
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.replace import replace
from ufl.argument import Coargument
from ufl.core.base_form_operator import BaseFormOperator, _split_dual_slot_arguments
from ufl.core.expr import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.form import BaseForm, Form, FormSum


def _replace_argument_slots(o: BaseFormOperator, mapping: dict) -> BaseFormOperator:
    """Replace the arguments in the argument slots of a base form operator.

    An argument inside an operand of ``o``, such as the Coargument of an
    operand Interpolate, is bound inside that operand. This function does not
    replace it, even if it is equal to a key of ``mapping``.

    Args:
        o: The base form operator.
        mapping: The replacements for the arguments in the argument slots of ``o``.

    Returns:
        The base form operator with the replaced argument slots.
    """
    slots = tuple(replace(slot, mapping) for slot in o.argument_slots())
    return o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=slots)


def _rank(o: Expr | BaseForm) -> int:
    """Return the form rank, treating coefficient expressions as rank one."""
    return len(o.arguments()) if isinstance(o, BaseForm) else 1


class BaseFormTraversal(DAGTraverser):
    """Traverse BaseForms, including operator operands and argument slots."""

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

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> Expr | BaseForm:
        """Visit an operator without treating its argument slots as operands."""
        return self._map_operator(o)

    def _map_operator(self, o: BaseFormOperator) -> BaseFormOperator:
        """Traverse an operator's operands and argument slots."""
        operands = tuple(self(op) for op in o.ufl_operands)
        slots = tuple(self(slot) for slot in o.argument_slots())
        if operands == o.ufl_operands and slots == o.argument_slots():
            return o
        return o._ufl_expr_reconstruct_(*operands, argument_slots=slots)

    @process.register(FormSum)
    @DAGTraverser.postorder
    def _(self, o: FormSum, *components: BaseForm) -> BaseForm:
        """Traverse the components of a FormSum."""
        return FormSum(*zip(components, o.weights()))

    @process.register(Adjoint)
    @DAGTraverser.postorder
    def _(self, o: Adjoint, form: BaseForm) -> Expr | BaseForm:
        """Reconstruct an Adjoint after traversing its form."""
        return Adjoint(form)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Reconstruct an Action after traversing its operands."""
        return Action(left, right)


class BaseFormRestructurer(BaseFormTraversal):
    """Split operator compositions and simplify Actions and Adjoints."""

    def __init__(
        self,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Cache the association and contraction traversers."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        # Each pass memoises a different transformation of the same nodes,
        # so their visited and result caches must remain independent.
        self._dag_traverser_cache: dict[type[BaseFormTraversal], BaseFormTraversal] = {
            ActionAssociator: ActionAssociator(compress=compress),
            ActionContractor: ActionContractor(self, compress=compress),
        }

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``."""
        return super().process(o)

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> Expr | BaseForm:
        """Split compositions in argument slots into Actions."""
        o = self._map_operator(o)
        v, *slots = o.argument_slots()
        operator_slots = [
            i
            for i, slot in enumerate(slots)
            if isinstance(slot, BaseFormOperator) and len(slot.arguments()) == 2
        ]
        if len(operator_slots) == 1:
            # A base form operator is linear in its other argument slots:
            # N(u; v*, B) -> Action(N(u; v*, uhat), B), where uhat is numbered
            # after the other arguments of N so that the Action contracts it
            # with the Coargument of B. An Action contracts one argument, so
            # an operator with more than one such slot is not split.
            (i,) = operator_slots
            B = slots[i]
            # The Coargument of B is bound inside B, so only the other
            # argument of B moves out of N.
            Bstar, *_ = B.argument_slots()
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

        dual = self(v)
        if len(dual.arguments()) > 1:
            # A base form operator is linear in its dual slot:
            # N(u; A) -> Action(A, N(u; v*)). Like the dual slot, the Action
            # contracts the highest-numbered argument of A, with the lowest of N(u; v*).
            contracted, free = _split_dual_slot_arguments(dual)
            primal_numbers = {a.number() for a in o.arguments() if a not in free}
            space = contracted.ufl_function_space().dual()
            if 0 in primal_numbers:
                # v* cannot be the lowest argument of N(u; v*), but the free
                # arguments of A avoid the primal numbers, so the contracted
                # argument can be the lowest of A instead:
                # N(u; A) -> Action(N(u; v*), A'), where v* is numbered last.
                vstar = contracted.reconstruct(function_space=space, number=1 + max(primal_numbers))
                left = o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(vstar, *slots))
                lowest = replace(dual, {contracted: contracted.reconstruct(number=0)})
                return self(Action(left, lowest))
            vstar = contracted.reconstruct(function_space=space, number=0)
            right = o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(vstar, *slots))
            return self(Action(dual, right))
        if dual is v:
            return o
        return o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(dual, *slots))

    @process.register(Adjoint)
    @DAGTraverser.postorder
    def _(self, o: Adjoint, form: BaseForm) -> Expr | BaseForm:
        """Swap argument numbers for the Adjoint of a base form operator."""
        if isinstance(form, BaseFormOperator):
            # The adjoint of a base form operator swaps the numbers of its arguments.
            u, v = form.arguments()
            adjoint = _replace_argument_slots(
                form, {u: u.reconstruct(number=v.number()), v: v.reconstruct(number=u.number())}
            )
            return self(adjoint)

        return Adjoint(form)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Restructure an Action after processing its operands."""
        if isinstance(left, Adjoint) and _rank(right) == 1:
            # Action(Adjoint(A), x) is the same contraction as Action(x, A).
            return self(Action(right, left.form()))

        associate = self._dag_traverser_cache[ActionAssociator]
        absorb = self._dag_traverser_cache[ActionContractor]
        return absorb(associate(Action(left, right)))


class ActionAssociator(BaseFormTraversal):
    """Associate Actions to contract rank-one operands first.

    Leave chains of rank-two operands unchanged, since their assembly cost
    depends on information unavailable here. Distribute over FormSums only
    to avoid assembling a sum of rank greater than one. Reassociate across
    rank-two operands only: for higher ranks, the argument contracted by an
    Action depends on its position in the chain.
    """

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``."""
        return super().process(o)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Associate an Action to avoid rank-two intermediate forms."""
        if _rank(left) == 1 and isinstance(right, Action) and _rank(right.left()) == 2:
            # A rank 1 operand is contracted first, so that every Action in the
            # chain gives another rank 1 operand:
            # Action(x, Action(A, B)) -> Action(Action(x, A), B)
            return self(Action(Action(left, right.left()), right.right()))

        if (
            isinstance(left, Action)
            and _rank(left.left()) > 1
            and _rank(left.right()) == 2
            and _rank(right) == 1
        ):
            # Action(Action(A, B), x) -> Action(A, Action(B, x)).
            return self(Action(left.left(), Action(left.right(), right)))

        if isinstance(left, FormSum) and _rank(left) > 1 and _rank(right) == 1:
            return FormSum(
                *((self(Action(c, right)), w) for c, w in zip(left.components(), left.weights()))
            )
        if isinstance(right, FormSum) and _rank(right) > 1 and _rank(left) == 1:
            # Action(L, A) contracts L with Adjoint(A), so the weights of
            # the rank-two FormSum A must be conjugated.
            return FormSum(
                *(
                    (self(Action(left, c)), w.conjugate() if isinstance(w, Complex) else Conj(w))
                    for c, w in zip(right.components(), right.weights())
                )
            )
        return Action(left, right)


class ActionContractor(BaseFormTraversal):
    """Contract arguments in Forms and BaseFormOperator slots."""

    def __init__(
        self,
        restructure: Callable[[Expr | BaseForm], Expr | BaseForm],
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Store the dispatcher used to restructure changed operators."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        # The dispatcher splits rank-two slots into Actions. Contractions fill
        # slots with rank-one forms or coefficient expressions, so restructuring
        # a contracted operator does not reverse those splits.
        self._restructure = restructure

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``."""
        return super().process(o)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Replace contracted arguments in forms and operator slots."""
        if (left is not o.left() and left != o.left()) or (
            right is not o.right() and right != o.right()
        ):
            # Completing a child contraction can expose a composition to this
            # parent. Redispatch only the Action whose operands changed.
            return self._restructure(Action(left, right))

        if (
            isinstance(right, BaseFormOperator)
            and isinstance(left, BaseForm)
            and len(left.arguments()) == 1
        ):
            # A base form operator is linear in its dual slot:
            # Action(L, N(u; v*, uhat)) -> N(u; L, uhat)
            vstar, *others = right.arguments()
            if right.argument_slots()[0] == vstar:
                # The other arguments are renumbered after the contracted v*.
                mapping = {a: a.reconstruct(number=a.number() - 1) for a in others}
                contracted = _replace_argument_slots(right, {vstar: left, **mapping})
                return self._restructure(contracted)

        if isinstance(left, BaseFormOperator):
            v, *slots = left.argument_slots()
            if isinstance(right, BaseForm) and len(right.arguments()) == 1:
                # A base form operator is linear in its dual slot:
                # Action(N(u; v*), R) -> N(u; R)
                if v == left.arguments()[-1]:
                    contracted = left._ufl_expr_reconstruct_(
                        *left.ufl_operands, argument_slots=(right, *slots)
                    )
                    return self._restructure(contracted)
            if isinstance(right, Expr) and _rank(right) == 1:
                # A base form operator is linear in its other argument slots:
                # Action(N(u; v*, uhat), w) -> N(u; v*, w)
                uhat = left.arguments()[-1]
                if v != uhat:
                    contracted = _replace_argument_slots(left, {uhat: right})
                    return self._restructure(contracted)

        if isinstance(left, Form) and isinstance(right, Expr) and _rank(right) == 1:
            return replace(left, {left.arguments()[-1]: right})
        if isinstance(right, Form) and isinstance(left, Expr) and _rank(left) == 1:
            # Contract the lowest-numbered argument, then conjugate the integrands.
            return map_integrands(Conj, replace(right, {right.arguments()[0]: left}))

        # Expression Sums cannot be assembled independently. Distribute over
        # these sums, keeping FormSums unexpanded.
        if isinstance(left, Sum):
            return FormSum(*((self(Action(c, right)), 1) for c in left.ufl_operands))
        if isinstance(right, Sum):
            return FormSum(*((self(Action(left, c)), 1) for c in right.ufl_operands))

        return Action(left, right)


def restructure_base_form(form: Expr | BaseForm, complex_mode: bool = False) -> Expr | BaseForm:
    """Expand derivatives and restructure a BaseForm for independent assembly.

    Leave Forms without base form operators unchanged for the form compiler.
    Split compositions in argument slots into Actions, associate rank-one
    contractions first, and replace contracted arguments in Forms and operator
    slots. Revisit only operators and Actions changed by those contractions.

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
