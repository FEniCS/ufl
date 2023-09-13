# -*- coding: utf-8 -*-
# Copyright (C) 2011-2017 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.classes import (CellAvg, FacetAvg, Grad, Indexed, NegativeRestricted,
                         PositiveRestricted, ReferenceGrad, ReferenceValue, Expr,
                         Terminal)
from ufl.corealg.map_dag import map_expr_dag
from functools import singledispatch

modifier_precedence = [
    ReferenceValue, ReferenceGrad, Grad, CellAvg, FacetAvg, PositiveRestricted,
    NegativeRestricted, Indexed
]

modifier_precedence = {
    m._ufl_handler_name_: i
    for i, m in enumerate(modifier_precedence)
}


def balance_modified_terminal(expr):
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


@singledispatch
def balance_modifiers(o, *ops):
    """Single-dispatch function to balance modifiers in an expression
    :arg o: the expression

    """
    raise AssertionError("UFL node expected, not %s" % type(o))


@balance_modifiers.register(Expr)
def balance_modifiers_expr(o, *ops):
    return o._ufl_expr_reconstruct_(*ops)


@balance_modifiers.register(Terminal)
def balance_modifiers_terminal(o, *ops):
    return o


@balance_modifiers.register(ReferenceValue)
@balance_modifiers.register(ReferenceGrad)
@balance_modifiers.register(Grad)
@balance_modifiers.register(CellAvg)
@balance_modifiers.register(FacetAvg)
@balance_modifiers.register(PositiveRestricted)
@balance_modifiers.register(NegativeRestricted)
def balance_modifiers_mod(o, *ops):
    return balance_modified_terminal(o)


def balance_modifiers(expr):
    mf = balance_modifiers
    return map_expr_dag(mf, expr)
