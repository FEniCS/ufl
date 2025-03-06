"""Remove component tensors.

This module contains classes and functions to remove component tensors.
"""
# Copyright (C) 2025 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import ComponentTensor, Index, MultiIndex, Zero
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction, memoized_handler


class IndexReplacer(MultiFunction):
    """Replace Indices."""

    def __init__(self, fimap: dict):
        """Initialise.

        Args:
           fimap: map for index replacements.

        """
        MultiFunction.__init__(self)
        self.fimap = fimap
        self._object_cache = {}

    expr = MultiFunction.reuse_if_untouched

    @memoized_handler
    def zero(self, o):
        """Handle Zero."""
        free_indices = []
        index_dimensions = []
        for i, d in zip(o.ufl_free_indices, o.ufl_index_dimensions):
            k = Index(i)
            j = self.fimap.get(k, k)
            if isinstance(j, Index):
                free_indices.append(j.count())
                index_dimensions.append(d)
        return Zero(
            shape=o.ufl_shape,
            free_indices=tuple(free_indices),
            index_dimensions=tuple(index_dimensions),
        )

    @memoized_handler
    def multi_index(self, o):
        """Handle MultiIndex."""
        return MultiIndex(tuple(self.fimap.get(i, i) for i in o.indices()))


class IndexRemover(MultiFunction):
    """Remove Indexed."""

    def __init__(self):
        """Initialise."""
        MultiFunction.__init__(self)
        self._object_cache = {}

    expr = MultiFunction.reuse_if_untouched

    @memoized_handler
    def _unary_operator(self, o):
        """Simplify UnaryOperator(Zero)."""
        (operand,) = o.ufl_operands
        f = map_expr_dag(self, operand)
        if isinstance(f, Zero):
            return Zero(
                shape=o.ufl_shape,
                free_indices=o.ufl_free_indices,
                index_dimensions=o.ufl_index_dimensions,
            )
        if f is operand:
            # Reuse if untouched
            return o
        return o._ufl_expr_reconstruct_(f)

    @memoized_handler
    def indexed(self, o):
        """Simplify Indexed."""
        o1, i1 = o.ufl_operands
        if isinstance(o1, ComponentTensor):
            # Simplify Indexed ComponentTensor
            o2, i2 = o1.ufl_operands
            # Replace inner indices first
            v = map_expr_dag(self, o2)
            # Replace outer indices
            # NOTE: Replace with `fimap = dict(zip(i2, i1, strict=True))` when
            # Python>=3.10
            assert len(i2) == len(i1)
            fimap = dict(zip(i2, i1))
            rule = IndexReplacer(fimap)
            return map_expr_dag(rule, v)

        expr = map_expr_dag(self, o1)
        if expr is o1:
            # Reuse if untouched
            return o
        return o._ufl_expr_reconstruct_(expr, i1)

    reference_grad = _unary_operator
    reference_value = _unary_operator


def remove_component_tensors(o):
    """Remove component tensors."""
    rule = IndexRemover()
    return map_integrand_dags(rule, o)
