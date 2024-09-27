"""Algorithms for renumbering of counted objects, currently variables and indices."""
# Copyright (C) 2008-2024 Martin Sandve Alnæs, Anders Logg, Jørgen S. Dokken and Lawerence Mitchell
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections import defaultdict
from itertools import count as _count

from ufl.classes import Form, Integral
from ufl.core.expr import Expr
from ufl.core.multiindex import Index
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction


class IndexRelabeller(MultiFunction):
    """Renumber indices to have a consistent index numbering starting from 0."""

    def __init__(self):
        """Initialize index relabeller with a zero count."""
        super().__init__()
        count = _count()
        self.index_cache = defaultdict(lambda: Index(next(count)))

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        """Apply to multi-indices."""
        return type(o)(
            tuple(self.index_cache[i] if isinstance(i, Index) else i for i in o.indices())
        )

    def zero(self, o):
        """Apply to zero."""
        fi = o.ufl_free_indices
        fid = o.ufl_index_dimensions
        new_indices = [self.index_cache[Index(i)].count() for i in fi]
        if fi == () and fid == ():
            return o
        new_fi, new_fid = zip(*sorted(zip(new_indices, fid), key=lambda x: x[0]))
        return type(o)(o.ufl_shape, tuple(new_fi), tuple(new_fid))


def renumber_indices(form):
    """Renumber indices to have a consistent index numbering starting from 0.

    This is useful to avoid multiple kernels for the same integrand,
    but with different subdomain ids.

    Args:
        form: A UFL form, integral or expression.

    Returns:
        A new form, integral or expression with renumbered indices.
    """
    if isinstance(form, Form):
        new_integrals = [renumber_indices(itg) for itg in form.integrals()]
        return Form(new_integrals)
    elif isinstance(form, Integral):
        integral = form
        reindexer = IndexRelabeller()
        new_integrand = map_expr_dag(reindexer, integral.integrand())
        return integral.reconstruct(new_integrand)
    elif isinstance(form, Expr):
        expr = form
        reindexer = IndexRelabeller()
        return map_expr_dag(reindexer, expr)
    else:
        raise ValueError(f"Invalid form type {form.__class__name}")
