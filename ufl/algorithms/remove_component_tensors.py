"""Remove component tensors.

This module contains classes and functions to remove component tensors.
"""
# Copyright (C) 2025 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from functools import singledispatchmethod

from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import ComponentTensor, Expr, Index, Indexed, MultiIndex, Zero
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.index_combination_utils import unique_sorted_indices


class IndexReplacer(DAGTraverser):
    """Replace Indices."""

    def __init__(self, fimap: dict):
        """Initialise.

        Args:
           fimap: map for index replacements.

        """
        super().__init__()
        self.fimap = fimap

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process an expression node."""
        return super().process(o)

    @process.register(Expr)
    @DAGTraverser.postorder
    def _(self, o: Expr, *ops: Expr) -> Expr:
        """Reuse an expression if none of its operands changed."""
        if all(a is b for a, b in zip(o.ufl_operands, ops)):
            return o
        return o._ufl_expr_reconstruct_(*ops)

    @process.register(Zero)
    def _(self, o: Zero) -> Zero:
        """Handle Zero."""
        indices = tuple(map(Index, o.ufl_free_indices))
        if not any(i in self.fimap for i in indices):
            # Reuse if untouched
            return o

        fi: list[tuple[int, int]] = []
        index_dimensions: tuple[int, ...] = o.ufl_index_dimensions
        for i, d in zip(indices, index_dimensions):
            j = self.fimap.get(i, i)
            if isinstance(j, Index):
                fi.append((j.count(), d))

        fi = unique_sorted_indices(sorted(fi))
        if fi:
            free_indices, index_dimensions = zip(*fi)
        else:
            free_indices, index_dimensions = (), ()

        return Zero(
            shape=o.ufl_shape,
            free_indices=free_indices,
            index_dimensions=index_dimensions,
        )

    @process.register(MultiIndex)
    def _(self, o: MultiIndex) -> MultiIndex:
        """Handle MultiIndex."""
        if not any(i in self.fimap for i in o):
            # Reuse if untouched
            return o

        indices = tuple(self.fimap.get(i, i) for i in o)
        return MultiIndex(indices)


class IndexRemover(DAGTraverser):
    """Remove Indexed."""

    def __init__(self):
        """Initialise."""
        super().__init__()
        self.rules = {}

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process an expression node."""
        return super().process(o)

    @process.register(Expr)
    @DAGTraverser.postorder
    def _(self, o: Expr, *ops: Expr) -> Expr:
        """Reuse an expression if none of its operands changed."""
        if all(a is b for a, b in zip(o.ufl_operands, ops)):
            return o
        return o._ufl_expr_reconstruct_(*ops)

    @process.register(Indexed)
    @DAGTraverser.postorder
    def _(self, o: Indexed, o1: Expr, i1: MultiIndex) -> Expr:
        """Simplify Indexed."""
        if isinstance(o1, ComponentTensor):
            # Simplify Indexed ComponentTensor
            o2, i2 = o1.ufl_operands
            # Replace outer indices
            rkey = (i2, i1)
            rule = self.rules.get(rkey)
            if rule is None:
                # NOTE: Replace with `fimap = dict(zip(i2, i1, strict=True))` when
                # Python>=3.10
                assert len(i2) == len(i1)
                fimap = dict(zip(i2, i1))
                rule = IndexReplacer(fimap)
                self.rules[rkey] = rule

            return rule(o2)

        elif o.ufl_operands[0] is o1:
            # Reuse if untouched
            return o
        else:
            return o._ufl_expr_reconstruct_(o1, i1)


def remove_component_tensors(o):
    """Remove component tensors."""
    rule = IndexRemover()
    return map_integrands(rule, o)
