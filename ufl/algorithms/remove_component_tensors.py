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

    def __init__(self, fimap: dict, object_cache=None):
        """Initialise.

        Args:
           fimap: map for index replacements.
           object_cache: a dict to cache objects.

        """
        MultiFunction.__init__(self)
        self.fimap = fimap
        self._object_cache = object_cache or {}
        # caches for reuse in the dispatched transformers
        self.vcaches = defaultdict(dict)
        self.rcaches = defaultdict(dict)

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

        key = (type(o), tuple(free_indices), tuple(index_dimensions))
        r = self._object_cache.get(key)
        if r is None:
            r = Zero(
                shape=o.ufl_shape,
                free_indices=tuple(free_indices),
                index_dimensions=tuple(index_dimensions),
            )
            self._object_cache[key] = r
        return r

    @memoized_handler
    def multi_index(self, o):
        """Handle MultiIndex."""
        if not any(i in self.fimap for i in o):
            # Reuse if untouched
            return o

        indices = tuple(self.fimap.get(i, i) for i in o)

        key = (type(o), *_cache_key(indices))
        r = self._object_cache.get(key)
        if r is None:
            r = MultiIndex(indices)
            self._object_cache[key] = r
        return r


class IndexRemover(MultiFunction):
    """Remove Indexed."""

    def __init__(self):
        """Initialise."""
        MultiFunction.__init__(self)
        self.rules = {}
        self.ocache = {}
        self.rcache = {}

    ufl_type = MultiFunction.reuse_if_untouched

    def indexed(self, o, o1, i1):
        """Simplify Indexed."""
        ckey = (o1, i1)
        result = self.rcache.get(ckey)
        if result is not None:
            return result

        if isinstance(o1, ComponentTensor):
            # Simplify Indexed ComponentTensor
            o2, i2 = o1.ufl_operands

            # Replace outer indices
            rkey = (_cache_key(i2), _cache_key(i1))
            rule = self.rules.get(rkey)
            if rule is None:
                # NOTE: Replace with `fimap = dict(zip(i2, i1, strict=True))` when
                # Python>=3.10
                assert len(i2) == len(i1)
                fimap = dict(zip(i2, i1))
                rule = IndexReplacer(fimap, object_cache=self.ocache)
                self.rules[rkey] = rule

            key = (IndexReplacer, o2)
            result = map_expr_dag(rule, o2, vcache=rule.vcaches[key], rcache=rule.rcaches[key])
            return self.rcache.setdefault(ckey, result)

        if o.ufl_operands[0] is o1:
            # Reuse if untouched
            result = o
        else:
            result = o._ufl_expr_reconstruct_(o1, i1)
        return self.rcache.setdefault(ckey, result)


def remove_component_tensors(o):
    """Remove component tensors."""
    rule = IndexRemover()
    return map_integrand_dags(rule, o)


def _cache_key(multiindex):
    """Return a cache key for a MultiIndex."""
    return tuple((type(j), j.count() if isinstance(j, Index) else int(j)) for j in multiindex)
