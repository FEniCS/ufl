"""Restructure BaseForms so that each operand can be assembled independently."""

# Copyright (C) 2026 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections.abc import Callable
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
from ufl.core.base_form_operator import BaseFormOperator, _number_contracted_argument
from ufl.core.expr import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.form import BaseForm, Form, FormSum


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
    """Split operator compositions, associate Actions and absorb their operands.

    Each Action goes through ActionAssociator, ActionAbsorber and ActionDistributor.
    The other rules are below, where ``N`` is an operator, ``B`` has rank two, ``A`` has
    rank above one, ``L`` is a one-form and ``w_i`` is an Argument numbered ``i``. An
    operator with one rank-two primal slot ``B`` is split first.

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

    def __init__(
        self,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Cache the association, absorption and distribution traversers."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        # Each pass memoises a different transformation of the same nodes,
        # so their visited and result caches must remain independent.
        self._dag_traverser_cache: dict[type[BaseFormTraversal], BaseFormTraversal] = {
            ActionAssociator: ActionAssociator(compress=compress),
            ActionAbsorber: ActionAbsorber(self, compress=compress),
            ActionDistributor: ActionDistributor(self, compress=compress),
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
        map_slots = [
            i
            for i, slot in enumerate(slots)
            if isinstance(slot, BaseForm) and len(slot.arguments()) == 2
        ]
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

        dual = self(v)
        if len(dual.arguments()) > 1:
            # A base form operator is linear in its dual slot, so N(u; A) splits into
            # Action(N(u; vhat), A) if the primal slots use number 0, and otherwise
            # into Action(A, N(u; vhat)). Each contracts the contracted argument of A.
            A = _number_contracted_argument(o, dual, o.arguments())
            slot_numbers = [a.number() for s in slots for a in extract_type(s, Argument, True)]
            operator_first = 0 in slot_numbers
            contracted = A.arguments()[0 if operator_first else -1]
            space = contracted.ufl_function_space().dual()
            vhat_number = 1 + max(slot_numbers) if operator_first else 0
            vhat = contracted.reconstruct(function_space=space, number=vhat_number)
            Nhat = o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(vhat, *slots))
            return self(Action(Nhat, A) if operator_first else Action(A, Nhat))
        if isinstance(dual, Form) and dual.arguments()[0].number() != 0:
            # The contracted argument of a one-form is bound in the dual slot, so its
            # number is free. Forms are assembled with arguments numbered from zero.
            (contracted,) = dual.arguments()
            dual = replace(dual, {contracted: contracted.reconstruct(number=0)})
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
            adjoint = replace(
                form, {u: u.reconstruct(number=v.number()), v: v.reconstruct(number=u.number())}
            )
            return self(adjoint)
        return Adjoint(form)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Restructure an Action after processing its operands."""
        associate = self._dag_traverser_cache[ActionAssociator]
        absorb = self._dag_traverser_cache[ActionAbsorber]
        distribute = self._dag_traverser_cache[ActionDistributor]
        return distribute(absorb(associate(Action(left, right))))


class ActionAssociator(BaseFormTraversal):
    """Associate Actions to contract rank-one operands first.

    Leave chains of rank-two operands unchanged, since their assembly cost
    depends on information unavailable here. Distribute over FormSums only
    to avoid assembling a sum of rank greater than one. Reassociate across
    rank-two operands only: for higher ranks, the argument contracted by an
    Action depends on its position in the chain.

    Here ``x`` has rank one, ``A`` and ``B`` have rank two except that the left operand
    of a chain can have a higher rank.

    ============================  ===================================================
    Expression                    Result
    ============================  ===================================================
    ``Action(Adjoint(A), x)``     ``Action(x, A)``
    ``Action(x, Action(A, B))``   ``Action(Action(x, A), B)``
    ``Action(Action(A, B), x)``   ``Action(A, Action(B, x))``
    ``Action(a * A + b * B, x)``  ``a * Action(A, x) + b * Action(B, x)``
    ``Action(x, a * A + b * B)``  ``conj(a) * Action(x, A) + conj(b) * Action(x, B)``
    ============================  ===================================================
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
        if isinstance(left, Adjoint) and _rank(right) == 1:
            # Action(Adjoint(A), x) is the same contraction as Action(x, A).
            return self(Action(right, left.form()))

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
                    (self(Action(left, c)), Conj(w))
                    for c, w in zip(right.components(), right.weights())
                )
            )
        return Action(left, right)


class ActionAbsorber(BaseFormTraversal):
    """Absorb Action operands into Forms and BaseFormOperator slots.

    The rules are tried in order, where ``N`` is an operator with ``v*`` or ``uhat`` free,
    ``L`` is a rank one BaseForm, ``F`` is a Form and ``w`` is an expression. An operator
    is restructured again after absorbing, as is an Action whose operands changed.

    =============================  ===================
    Expression                     Result
    =============================  ===================
    ``Action(L, N(u; v*, uhat))``  ``N(u; L, uhat)``
    ``Action(N(u; v*), L)``        ``N(u; L)``
    ``Action(N(u; v*, uhat), w)``  ``N(u; v*, w)``
    ``Action(F(v_0, v_1), w)``     ``F(v_0, w)``
    ``Action(w, F(v_0, v_1))``     ``conj(F(w, v_0))``
    =============================  ===================
    """

    def __init__(
        self,
        restructure: Callable[[Expr | BaseForm], Expr | BaseForm],
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Store the dispatcher used to restructure changed operators."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        # The dispatcher splits rank-two slots into Actions. Absorption fills
        # slots with rank-one forms or coefficient expressions, so restructuring
        # an operator after absorption does not reverse those splits.
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
        """Absorb an operand of an Action into a form or operator slot."""
        if (left is not o.left() and left != o.left()) or (
            right is not o.right() and right != o.right()
        ):
            # Absorbing into a child can expose a composition to this
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
                absorbed = replace(right, {vstar: left, **mapping})
                return self._restructure(absorbed)

        if isinstance(left, BaseFormOperator):
            v, *slots = left.argument_slots()
            if isinstance(right, BaseForm) and len(right.arguments()) == 1:
                # A base form operator is linear in its dual slot:
                # Action(N(u; v*), R) -> N(u; R)
                if v == left.arguments()[-1]:
                    absorbed = left._ufl_expr_reconstruct_(
                        *left.ufl_operands, argument_slots=(right, *slots)
                    )
                    return self._restructure(absorbed)
            if isinstance(right, Expr) and _rank(right) == 1:
                # A base form operator is linear in its other argument slots:
                # Action(N(u; v*, uhat), w) -> N(u; v*, w)
                uhat = left.arguments()[-1]
                if v != uhat:
                    absorbed = replace(left, {uhat: right})
                    return self._restructure(absorbed)

        if isinstance(left, Form) and isinstance(right, Expr) and _rank(right) == 1:
            return replace(left, {left.arguments()[-1]: right})
        if isinstance(right, Form) and isinstance(left, Expr) and _rank(left) == 1:
            # Contract the lowest-numbered argument, then conjugate the integrands.
            # The other arguments are renumbered after the contracted argument.
            contracted, *others = right.arguments()
            mapping = {a: a.reconstruct(number=a.number() - 1) for a in others}
            return map_integrands(Conj, replace(right, {contracted: left, **mapping}))

        return Action(left, right)


class ActionDistributor(BaseFormTraversal):
    """Distribute Actions over the expression Sums that were not absorbed.

    An expression Sum cannot be assembled independently, unlike a FormSum, which stays
    unexpanded. Here ``a + b`` is an expression Sum.

    ====================  ===============================
    Expression            Result
    ====================  ===============================
    ``Action(a + b, R)``  ``Action(a, R) + Action(b, R)``
    ``Action(L, a + b)``  ``Action(L, a) + Action(L, b)``
    ====================  ===============================
    """

    def __init__(
        self,
        restructure: Callable[[Expr | BaseForm], Expr | BaseForm],
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Store the dispatcher that restructures the distributed Actions."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
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
        """Distribute an Action over an expression Sum in an operand."""
        if isinstance(left, Sum):
            return FormSum(*((self._restructure(Action(c, right)), 1) for c in left.ufl_operands))
        if isinstance(right, Sum):
            return FormSum(*((self._restructure(Action(left, c)), 1) for c in right.ufl_operands))
        if (left is not o.left() and left != o.left()) or (
            right is not o.right() and right != o.right()
        ):
            # Distributing in a child can expose a composition to this parent.
            return self._restructure(Action(left, right))
        return o


def restructure_base_form(form: Expr | BaseForm, complex_mode: bool = False) -> Expr | BaseForm:
    """Expand derivatives and restructure a BaseForm for independent assembly.

    Leave Forms without base form operators unchanged for the form compiler.
    Split compositions in argument slots into Actions, associate rank-one
    contractions first, absorb Action operands into Forms and operator slots,
    and distribute over the remaining expression Sums. Revisit only changed nodes.

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
