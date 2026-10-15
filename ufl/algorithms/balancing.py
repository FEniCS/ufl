"""Balancing."""
# -*- coding: utf-8 -*-
# Copyright (C) 2011-2017 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from functools import singledispatchmethod

from ufl.classes import (
    Expr,
    Grad,
    Indexed,
    Interpolate,
    NegativeRestricted,
    PositiveRestricted,
    ReferenceGrad,
    ReferenceValue,
    Terminal,
)
from ufl.corealg.dag_traverser import DAGTraverser

modifier_precedence = {
    m._ufl_handler_name_: i
    for i, m in enumerate(
        [
            ReferenceValue,
            ReferenceGrad,
            Grad,
            PositiveRestricted,
            NegativeRestricted,
            Indexed,
        ]
    )
}


def _is_modifiable(expr):
    """Check whether modifiers may be applied to ``expr``.

    An interpolation is modifiable like a terminal: it is a finite element field
    of its target space, and the expression it interpolates is evaluated
    separately.
    """
    return expr._ufl_is_terminal_ or isinstance(expr, Interpolate)


def balance_modified_terminal(expr):
    """Balance modified terminal."""
    # NB! Assuming e.g. grad(cell_avg(expr)) does not occur,
    # i.e. it is simplified to 0 immediately.
    if _is_modifiable(expr):
        return expr

    assert expr._ufl_is_terminal_modifier_

    orig = expr

    # Build list of modifier layers
    layers = [expr]
    while not _is_modifiable(expr):
        assert expr._ufl_is_terminal_modifier_
        expr = expr.ufl_operands[0]
        layers.append(expr)
    assert layers[-1] is expr
    assert _is_modifiable(expr)

    # Apply modifiers in order
    layers = sorted(layers[:-1], key=lambda e: modifier_precedence[e._ufl_handler_name_])
    for op in layers:
        ops = (expr,) + op.ufl_operands[1:]
        expr = op._ufl_expr_reconstruct_(*ops)

    # Preserve id if nothing has changed
    return orig if expr == orig else expr


class BalanceModifiers(DAGTraverser):
    """Balance modifiers."""

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process an expression node."""
        return super().process(o)

    @process.register(Expr)
    @DAGTraverser.postorder
    def _(self, o: Expr, *ops: Expr) -> Expr:
        """Apply to an expression."""
        return o._ufl_expr_reconstruct_(*ops)

    @process.register(Terminal)
    def _(self, expr: Terminal) -> Terminal:
        """Apply to terminal."""
        return expr

    @process.register(ReferenceValue)
    @process.register(ReferenceGrad)
    @process.register(Grad)
    @process.register(PositiveRestricted)
    @process.register(NegativeRestricted)
    @DAGTraverser.postorder
    def _(self, expr: Expr, *ops: Expr) -> Expr:
        """Apply to modifier."""
        return balance_modified_terminal(expr)


def balance_modifiers(expr):
    """Balance modifiers."""
    return BalanceModifiers()(expr)
