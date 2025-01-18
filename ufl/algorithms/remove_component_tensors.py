"""Remove component tensors.

This module contains classes and functions to remove component tensors.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.classes import (
    ComponentTensor,
    Form,
    Index,
    MultiIndex,
    Zero,
)
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
            if Index(i) in self.fimap:
                ind_j = self.fimap[Index(i)]
                if isinstance(ind_j, Index):
                    free_indices.append(ind_j.count())
                    index_dimensions.append(d)
            else:
                free_indices.append(i)
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
    def _zero_simplify(self, o):
        """Apply simplification for Zero()."""
        (operand,) = o.ufl_operands
        operand = map_expr_dag(self, operand)
        if isinstance(operand, Zero):
            return Zero(
                shape=o.ufl_shape,
                free_indices=o.ufl_free_indices,
                index_dimensions=o.ufl_index_dimensions,
            )
        return o._ufl_expr_reconstruct_(operand)

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
            assert len(i2) == len(i1)
            fimap = dict(zip(i2, i1))
            rule = IndexReplacer(fimap)
            return map_expr_dag(rule, v)

        expr = map_expr_dag(self, o1)
        if expr is o1:
            # Reuse if untouched
            return o
        return o._ufl_expr_reconstruct_(expr, i1)

    # Do something nicer
    positive_restricted = _zero_simplify
    negative_restricted = _zero_simplify
    reference_grad = _zero_simplify
    reference_value = _zero_simplify


def remove_component_tensors(o):
    """Remove component tensors."""
    if isinstance(o, Form):
        integrals = []
        for integral in o.integrals():
            integrand = remove_component_tensors(integral.integrand())
            if not isinstance(integrand, Zero):
                integrals.append(integral.reconstruct(integrand=integrand))
        return o._ufl_expr_reconstruct_(integrals)
    else:
        rule = IndexRemover()
        return map_expr_dag(rule, o)
