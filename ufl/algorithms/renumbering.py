# -*- coding: utf-8 -*-
"Algorithms for renumbering of counted objects, currently variables and indices."

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.log import error
from ufl.core.expr import Expr
from ufl.core.multiindex import Index, FixedIndex, MultiIndex
from ufl.variable import Label, Variable
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.classes import Zero


class VariableRenumberingTransformer(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)
        self.variable_map = {}

    def variable(self, o):
        e, l = o.ufl_operands  # noqa: E741
        v = self.variable_map.get(l)
        if v is None:
            e = self.visit(e)
            l2 = Label(len(self.variable_map))
            v = Variable(e, l2)
            self.variable_map[l] = v
        return v


class IndexRenumberingTransformer(VariableRenumberingTransformer):
    "This is a poorly designed algorithm. It is used in some tests, please do not use for anything else."

    def __init__(self):
        VariableRenumberingTransformer.__init__(self)
        self.index_map = {}

    def zero(self, o):
        fi = o.ufl_free_indices
        fid = o.ufl_index_dimensions
        mapped_fi = tuple(self.index(Index(count=i)) for i in fi)
        paired_fid = [(mapped_fi[pos], fid[pos]) for pos, a in enumerate(fi)]
        new_fi, new_fid = zip(*tuple(sorted(paired_fid)))
        return Zero(o.ufl_shape, new_fi, new_fid)

    def index(self, o):
        if isinstance(o, FixedIndex):
            return o
        else:
            c = o._count
            i = self.index_map.get(c)
            if i is None:
                i = Index(count=len(self.index_map))
                self.index_map[c] = i
            return i

    def multi_index(self, o):
        new_indices = tuple(self.index(i) for i in o.indices())
        return MultiIndex(new_indices)


def renumber_indices(expr):
    if isinstance(expr, Expr):
        num_free_indices = len(expr.ufl_free_indices)

    result = apply_transformer(expr, IndexRenumberingTransformer())

    if isinstance(expr, Expr):
        if num_free_indices != len(result.ufl_free_indices):
            error("The number of free indices left in expression should be invariant w.r.t. renumbering.")
    return result
