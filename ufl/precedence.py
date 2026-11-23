"""Precedence handling."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import warnings

# FIXME: This code is crap...


def parstr(child, parent, pre="(", post=")", format=str):
    """Parstr."""
    # Execute when needed instead of on import, which leads to all
    # kinds of circular trouble.  Fixing this could be an optimization
    # of str(expr) though.
    if not hasattr(parent, "_precedence"):
        assign_precedences(build_precedence_list())

    # We want child to be evaluated fully first, and if the parent has
    # higher precedence we later wrap in ().
    s = format(child)

    # Operators where operands are always parenthesized because
    # precedence is not defined below
    if parent._precedence == 0:
        return pre + s + post

    # If parent operator binds stronger than child, must parenthesize
    # child
    # FIXME: Is this correct for all possible positions of () in a + b + c?
    # FIXME: Left-right rule
    if parent._precedence > child._precedence:  # parent = indexed, child = terminal
        return pre + s + post

    # Nothing needed
    return s


def build_precedence_list():
    """Build precedence list."""
    from ufl.classes import (
        Abs,
        BesselFunction,
        Division,
        Indexed,
        IndexSum,
        MathFunction,
        Operator,
        Power,
        Product,
        Sum,
        Terminal,
    )

    # TODO: Fill in other types...
    # Power <= Transposed

    precedence_list = []
    # Default operator behaviour: should always add parentheses
    precedence_list.append((Operator,))

    precedence_list.append((Sum,))

    # sum_i a + b = (sum_i a) + b != sum_i (a + b), sum_i binds
    # stronger than +, but weaker than product
    precedence_list.append((IndexSum,))

    precedence_list.append(
        (
            Product,
            Division,
        )
    )

    # NB! Depends on language!
    precedence_list.append((Power, MathFunction, BesselFunction, Abs))

    precedence_list.append((Indexed,))

    # Default terminal behaviour: should never add parentheses
    precedence_list.append((Terminal,))
    return precedence_list


def build_precedence_mapping(precedence_list):
    """Given a precedence list, build a dict with class->int mappings.

    Each registered expression type gets the precedence of the closest class
    in its MRO that appears in the list. Types without one are returned as
    missing.

    Utility function used by some external code.
    """
    from ufl.core.expr import Expr
    from ufl.core.ufl_type import registry

    # Assign integer values for each precedence level
    levels = {c: k for k, p in enumerate(precedence_list) for c in p}
    pm = dict(levels)
    missing = set()
    for c in registry.classes:
        if c is Expr or not issubclass(c, Expr) or c in pm:
            continue
        level = next((levels[b] for b in c.__mro__ if b in levels), None)
        if level is None:
            missing.add(c)
        else:
            pm[c] = level
    return pm, missing


def assign_precedences(precedence_list):
    """Given a precedence list, assign ints to class._precedence."""
    pm, missing = build_precedence_mapping(precedence_list)
    for c, p in sorted(pm.items(), key=lambda x: x[0].__name__):
        c._precedence = p
    if missing:
        warnings.warn(
            "Missing precedence levels for classes:\n"
            + "\n".join(f"  {c}" for c in sorted(missing, key=lambda c: c.__name__))
        )
