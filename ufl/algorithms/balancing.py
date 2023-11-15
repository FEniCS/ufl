"""Balancing."""
# -*- coding: utf-8 -*-
# Copyright (C) 2011-2017 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.classes import (CellAvg, FacetAvg, Grad, Indexed, NegativeRestricted, PositiveRestricted, ReferenceGrad,
                         ReferenceValue)
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction

modifier_precedence = [
    ReferenceValue, ReferenceGrad, Grad, CellAvg, FacetAvg, PositiveRestricted,
    NegativeRestricted, Indexed
]

modifier_precedence = {
    m._ufl_handler_name_: i
    for i, m in enumerate(modifier_precedence)
}


def balance_modified_terminal(expr):
    """Balance modified terminal."""
    # NB! Assuming e.g. grad(cell_avg(expr)) does not occur,
    # i.e. it is simplified to 0 immediately.

    if expr._ufl_is_terminal_:
        return expr

    assert expr._ufl_is_terminal_modifier_

    orig = expr

    # Build list of modifier layers
    layers = [expr]
    while not expr._ufl_is_terminal_:
        assert expr._ufl_is_terminal_modifier_
        expr = expr.ufl_operands[0]
        layers.append(expr)
    assert layers[-1] is expr
    assert expr._ufl_is_terminal_

    # Apply modifiers in order
    layers = sorted(
        layers[:-1], key=lambda e: modifier_precedence[e._ufl_handler_name_])
    for op in layers:
        ops = (expr, ) + op.ufl_operands[1:]
        expr = op._ufl_expr_reconstruct_(*ops)

    # Preserve id if nothing has changed
    return orig if expr == orig else expr


class BalanceModifiers(MultiFunction):
    """Balance modifiers."""

    def expr(self, expr, *ops):
        """Apply to expr."""
        return expr._ufl_expr_reconstruct_(*ops)

    def terminal(self, expr):
        """Apply to terminal."""
        return expr

    def _modifier(self, expr, *ops):
        """Apply to _modifier."""
        return balance_modified_terminal(expr)

    reference_value = _modifier
    reference_grad = _modifier
    grad = _modifier
    cell_avg = _modifier
    facet_avg = _modifier
    positive_restricted = _modifier
    negative_restricted = _modifier


def balance_modifiers(expr):
    """Balance modifiers."""
    mf = BalanceModifiers()
    return map_expr_dag(mf, expr)
