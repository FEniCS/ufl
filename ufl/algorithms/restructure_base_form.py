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
from ufl.algorithms.replace import replace
from ufl.coefficient import Coefficient
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.expr import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.form import BaseForm, FormSum


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
            return replace(
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
                return replace(right, {vstar: left, **mapping})

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
                    return replace(left, {uhat: right})

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
