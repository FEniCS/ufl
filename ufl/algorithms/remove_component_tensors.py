"""Remove component tensors.

This module contains classes and functions to remove component tensors.
"""
# Copyright (C) 2025 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections import defaultdict

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import ComponentTensor, Index, MultiIndex, Zero
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.index_combination_utils import unique_sorted_indices


class IndexReplacer(MultiFunction):
    """Replace Indices."""

    def __init__(self, fimap: dict):
        """Initialise.

        Args:
           fimap: map for index replacements.

        """
        MultiFunction.__init__(self)
        self.fimap = fimap

    expr = MultiFunction.reuse_if_untouched

    def zero(self, o):
        """Handle Zero."""
        indices = tuple(map(Index, o.ufl_free_indices))
        if not any(i in self.fimap for i in indices):
            # Reuse if untouched
            return o

        fi = []
        for i, d in zip(indices, o.ufl_index_dimensions):
            j = self.fimap.get(i, i)
            if isinstance(j, Index):
                fi.append((j.count(), d))

        fi = unique_sorted_indices(sorted(fi))
        free_indices, index_dimensions = zip(*fi)

        return Zero(
            shape=o.ufl_shape,
            free_indices=free_indices,
            index_dimensions=index_dimensions,
        )

    def multi_index(self, o):
        """Handle MultiIndex."""
        if not any(i in self.fimap for i in o):
            # Reuse if untouched
            return o

        indices = tuple(self.fimap.get(i, i) for i in o)
        return MultiIndex(indices)


class IndexRemover(MultiFunction):
    """Remove Indexed."""

    def __init__(self):
        """Initialise."""
        MultiFunction.__init__(self)
        self.rules = {}
        # caches for reuse in the dispatched transformers
        self.vcaches = defaultdict(dict)
        self.rcaches = defaultdict(dict)

    expr = MultiFunction.reuse_if_untouched

    def indexed(self, o, o1, i1):
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

            key = (IndexReplacer, *rkey)
            return map_expr_dag(rule, o2, vcache=self.vcaches[key], rcache=self.rcaches[key])

        elif o.ufl_operands[0] is o1:
            # Reuse if untouched
            return o
        else:
            return o._ufl_expr_reconstruct_(o1, i1)


def remove_component_tensors(o):
    """Remove component tensors."""
    rule = IndexRemover()
    return map_integrand_dags(rule, o)
