"""Algorithm for replacing terminals in an expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

from collections import OrderedDict
from functools import singledispatchmethod

import ufl.classes
from ufl.algorithms.analysis import extract_type, has_exact_type
from ufl.algorithms.map_integrands import map_integrands
from ufl.argument import Argument, Coargument
from ufl.classes import BaseForm, CoefficientDerivative
from ufl.constantvalue import Zero, as_ufl
from ufl.core.base_form_operator import _split_dual_slot_arguments
from ufl.core.expr import Expr
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.form import ZeroBaseForm


class Replacer(DAGTraverser):
    """Replacer."""

    def __init__(self, mapping):
        """Initialize."""
        # Preserve the identity of mapped objects; compression may substitute
        # an equal but different expression.
        super().__init__(compress=False)
        self.mapping = mapping

        # One can replace Coarguments by 1-Forms
        def get_shape(x):
            """Get the shape of an object."""
            if isinstance(x, BaseForm) and not isinstance(x, Expr):
                arguments = x.arguments()
                return arguments[0].ufl_shape if arguments else ()
            return x.ufl_shape

        if not all(get_shape(k) == get_shape(v) for k, v in mapping.items()):
            raise ValueError(
                "Replacement expressions must have the same shape as what they replace."
            )

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr | ufl.classes.BaseForm):
        """Process ``o``."""
        return super().process(o)

    @process.register(ufl.classes.Expr)
    @process.register(ufl.classes.BaseForm)
    @DAGTraverser.postorder
    def _(self, o, *args):
        """Replace a ufl_type."""
        try:
            return self.mapping[o]
        except KeyError:
            if all(new is old for new, old in zip(args, o.ufl_operands)):
                return o
            return o._ufl_expr_reconstruct_(*args)

    @process.register(ufl.classes.BaseFormOperator)
    def _(self, o):
        """Replace a base form operator.

        The Coargument of an operator in an operand or a primal slot of ``o`` pairs
        with the output of that operator, so it is bound there. Only the dual slot of
        ``o`` can hold a free Coargument.
        """
        if o in self.mapping:
            return self.mapping[o]
        primal_mapping = {k: v for k, v in self.mapping.items() if not isinstance(k, Coargument)}
        vstar, *slots = o.argument_slots()
        new_ops = tuple(replace(op, primal_mapping) for op in o.ufl_operands)
        new_slots = (
            replace(vstar, self.mapping),
            *(replace(slot, primal_mapping) for slot in slots),
        )
        if new_ops == o.ufl_operands and new_slots == o.argument_slots():
            return o
        new = o._ufl_expr_reconstruct_(*new_ops, argument_slots=new_slots)
        if not any(isinstance(slot, Zero) for slot in new_slots[1:]):
            return new
        # o is linear in its primal slots.
        if new._parent_type is not BaseForm:
            return Zero(o.ufl_shape)
        # The dual slot contracts o into a form, which keeps the Arguments of the
        # slots that became zero.
        mapped = (primal_mapping.get(a, a) for s in slots for a in extract_type(s, Argument, True))
        slot_arguments = [a for a in mapped if isinstance(a, Argument)]
        _, dual_arguments = _split_dual_slot_arguments(new_slots[0], slot_arguments)
        arguments = OrderedDict.fromkeys((*dual_arguments, *slot_arguments))
        return ZeroBaseForm(tuple(sorted(arguments, key=lambda a: a.number())))

    @process.register(ufl.classes.CoefficientDerivative)
    def _(self, o):
        """Replace a coefficient derivative."""
        raise ValueError("Derivatives should be applied before executing replace.")

    @process.register(ufl.classes.BaseFormDerivative)
    def _(self, o):
        """Replace a base form derivative.

        The replacement acts on the result of the derivative. If it acted on the
        differentiated base form instead, then derivative(f(g), g) with g mapped to
        an expression would differentiate with respect to that expression.
        """
        if o in self.mapping:
            return self.mapping[o]
        # Hack to avoid circular dependencies
        from ufl.algorithms.ad import expand_coefficient_derivatives

        return replace(expand_coefficient_derivatives(o), self.mapping)


def replace(e, mapping):
    """Replace subexpressions in expression.

    Params:
        e: An Expr or Form
        mapping: A dict with from:to replacements to perform

    Returns:
        The expression with replacements performed
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.items())

    # Workaround for problem with delayed derivative evaluation
    # The problem is that J = derivative(f(g, h), g) does not evaluate immediately
    # So if we subsequently do replace(J, {g: h}) we end up with an expression:
    # derivative(f(h, h), h)
    # rather than what were were probably thinking of:
    # replace(derivative(f(g, h), g), {g: h})
    #
    # To fix this would require one to expand derivatives early (which
    # is not attractive), or make replace lazy too.
    if has_exact_type(e, CoefficientDerivative):
        # Hack to avoid circular dependencies
        from ufl.algorithms.ad import expand_coefficient_derivatives

        e = expand_coefficient_derivatives(e)

    return map_integrands(lambda expr: Replacer(mapping2)(expr), e)
