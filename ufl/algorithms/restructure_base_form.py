"""Algorithm for restructuring a BaseForm into an assembleable DAG."""

# Copyright (C) 2026 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from functools import singledispatchmethod

from ufl.action import Action
from ufl.adjoint import Adjoint
from ufl.algebra import Sum
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.replace import replace
from ufl.argument import Argument, Coargument
from ufl.coefficient import Coefficient
from ufl.core.base_form_operator import BaseFormOperator, _renumber_dual_slot_arguments
from ufl.core.expr import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.form import BaseForm, FormSum


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


class BaseFormRestructurer(DAGTraverser):
    """Simplify Actions and Adjoints so that each operand can be assembled on its own."""

    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``."""
        return super().process(o)

    @process.register(Expr)
    @process.register(BaseForm)
    def _(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Leave any other node unchanged."""
        return o

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> Expr | BaseForm:
        """Restructure the argument slots of a base form operator."""
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
            return self.action(self(left), self(B))

        if isinstance(v, Coargument):
            return o

        dual = self(v)
        if len(dual.arguments()) > 1:
            # A base form operator is linear in its dual slot:
            # N(u; A) -> Action(N(u; v*), A), where v* is numbered after the
            # arguments of the other slots so that the Action contracts it with
            # the first argument of A.
            primal_args = tuple(
                a for slot in slots for a in extract_type(slot, Argument, base_form_op_as_expr=True)
            )
            dual_slot_args = _renumber_dual_slot_arguments(v, (a.number() for a in primal_args))
            others = (a for a in o.arguments() if a not in dual_slot_args)
            number = max((a.number() for a in others), default=-1) + 1
            test = dual.arguments()[0]
            vstar = test.reconstruct(function_space=test.ufl_function_space().dual(), number=number)
            left = o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(vstar, *slots))
            return self.action(left, dual)
        if dual is v:
            return o
        return o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(dual, *slots))

    @process.register(FormSum)
    @DAGTraverser.postorder
    def _(self, o: FormSum, *components: BaseForm) -> BaseForm:
        """Restructure the components of a FormSum."""
        return FormSum(*zip(components, o.weights()))

    @process.register(Adjoint)
    @DAGTraverser.postorder
    def _(self, o: Adjoint, form: BaseForm) -> Expr | BaseForm:
        """Restructure an Adjoint."""
        return self.adjoint(form)

    @process.register(Action)
    @DAGTraverser.postorder
    def _(self, o: Action, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Restructure an Action."""
        return self.action(left, right)

    def adjoint(self, form: BaseForm) -> Expr | BaseForm:
        """Restructure the Adjoint of a restructured form."""
        if isinstance(form, FormSum):
            return FormSum(
                *((self.adjoint(c), w) for c, w in zip(form.components(), form.weights()))
            )

        if isinstance(form, BaseFormOperator):
            # The adjoint of a base form operator swaps the numbers of its arguments.
            u, v = form.arguments()
            return _replace_argument_slots(
                form, {u: u.reconstruct(number=v.number()), v: v.reconstruct(number=u.number())}
            )

        return Adjoint(form)

    def action(self, left: Expr | BaseForm, right: Expr | BaseForm) -> Expr | BaseForm:
        """Restructure the Action of restructured operands."""
        if isinstance(left, Adjoint) and (
            isinstance(right, Coefficient)
            or (isinstance(right, BaseForm) and len(right.arguments()) == 1)
        ):
            # Action(Adjoint(A), x) is the same contraction as Action(x, A).
            return self.action(right, left.form())

        if (
            isinstance(left, Action)
            and isinstance(left.right(), BaseForm)
            and len(left.right().arguments()) > 1
        ):
            # Action(Action(A, B), C) -> Action(A, Action(B, C))
            return self.action(left.left(), self.action(left.right(), right))

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
                return _replace_argument_slots(right, {vstar: left, **mapping})

        if isinstance(left, BaseFormOperator):
            v, *slots = left.argument_slots()
            if isinstance(right, BaseForm) and len(right.arguments()) == 1:
                # A base form operator is linear in its dual slot:
                # Action(N(u; v*), R) -> N(u; R)
                if v == left.arguments()[-1]:
                    return left._ufl_expr_reconstruct_(
                        *left.ufl_operands, argument_slots=(right, *slots)
                    )
            if isinstance(right, Coefficient):
                # A base form operator is linear in its other argument slots:
                # Action(N(u; v*, uhat), w) -> N(u; v*, w)
                uhat = left.arguments()[-1]
                if v != uhat:
                    return _replace_argument_slots(left, {uhat: right})

        # Otherwise, Action distributes over sums, so that the rules above
        # apply to each term. This enlarges the DAG, so it is the last resort.
        if isinstance(left, Sum):
            return FormSum(*((self.action(c, right), 1) for c in left.ufl_operands))
        if isinstance(left, FormSum):
            return FormSum(
                *((self.action(c, right), w) for c, w in zip(left.components(), left.weights()))
            )
        if isinstance(right, Sum):
            return FormSum(*((self.action(left, c), 1) for c in right.ufl_operands))
        if isinstance(right, FormSum):
            return FormSum(
                *((self.action(left, c), w) for c, w in zip(right.components(), right.weights()))
            )

        return Action(left, right)


def restructure_base_form(form: Expr | BaseForm) -> Expr | BaseForm:
    """Simplify the Actions and Adjoints of a BaseForm into an assembleable DAG.

    Args:
        form: The BaseForm, with its derivatives expanded.

    Returns:
        The restructured BaseForm.
    """
    return BaseFormRestructurer()(form)
