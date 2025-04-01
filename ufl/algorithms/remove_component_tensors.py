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
        indices = tuple(map(Index, o.ufl_free_indices))
        if not any(i in self.fimap for i in indices):
            # Reuse if untouched
            return o

        free_indices = []
        index_dimensions = []
        for i, d in zip(indices, o.ufl_index_dimensions):
            j = self.fimap.get(i, i)
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
        if any(i in self.fimap for i in o):
            indices = tuple(self.fimap.get(i, i) for i in o)
            key = _cache_key(indices)
            r = self._object_cache.get(key)
            if r is None:
                r = MultiIndex(indices)
                self._object_cache[key] = r
            return r

        # Reuse if untouched
        return o


class IndexRemover(MultiFunction):
    """Remove Indexed."""

    def __init__(self):
        """Initialise."""
        MultiFunction.__init__(self)
        # caches for reuse in the dispatched transformers
        self.vcaches = defaultdict(dict)
        self.rcaches = defaultdict(dict)
        self.rules = {}

    ufl_type = MultiFunction.reuse_if_untouched

    @memoized_handler
    def indexed(self, o):
        """Simplify Indexed."""
        o1, i1 = o.ufl_operands
        if isinstance(o1, ComponentTensor):
            # Simplify Indexed ComponentTensor
            o2, i2 = o1.ufl_operands

            # Remove inner indices first
            key = (IndexRemover, o2)
            v = map_expr_dag(self, o2, vcache=self.vcaches[key], rcache=self.rcaches[key])

            # Replace outer indices
            rkey = (IndexReplacer, _cache_key(i2), _cache_key(i1))
            try:
                rule = self.rules[rkey]
            except KeyError:
                # NOTE: Replace with `fimap = dict(zip(i2, i1, strict=True))` when
                # Python>=3.10
                assert len(i2) == len(i1)
                fimap = dict(zip(i2, i1))
                rule = IndexReplacer(fimap)
                self.rules.setdefault(rkey, rule)

            key = (*rkey, v)
            return map_expr_dag(rule, v, vcache=self.vcaches[key], rcache=self.rcaches[key])

        key = (IndexRemover, o1)
        expr = map_expr_dag(self, o1, vcache=self.vcaches[key], rcache=self.rcaches[key])
        if expr is o1:
            # Reuse if untouched
            return o
        return o._ufl_expr_reconstruct_(expr, i1)


def remove_component_tensors(o):
    """Remove component tensors."""
    rule = IndexRemover()
    o = map_integrand_dags(rule, o)
    return o


def _cache_key(multiindex):
    """Return a cache key for a MultiIndex."""
    return tuple((type(j), j.count() if isinstance(j, Index) else int(j)) for j in multiindex)
