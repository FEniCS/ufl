# -*- coding: utf-8 -*-
"""This module defines the Indexed class."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl_legacy.log import error
from ufl_legacy.constantvalue import Zero
from ufl_legacy.core.expr import Expr, ufl_err_str
from ufl_legacy.core.ufl_type import ufl_type
from ufl_legacy.core.operator import Operator
from ufl_legacy.core.multiindex import Index, FixedIndex, MultiIndex
from ufl_legacy.index_combination_utils import unique_sorted_indices, merge_unique_indices
from ufl_legacy.precedence import parstr


# --- Indexed expression ---

@ufl_type(is_shaping=True, num_ops=2, is_terminal_modifier=True)
class Indexed(Operator):
    __slots__ = (
        "ufl_free_indices",
        "ufl_index_dimensions",
    )

    def __new__(cls, expression, multiindex):
        if isinstance(expression, Zero):
            # Zero-simplify indexed Zero objects
            shape = expression.ufl_shape
            efi = expression.ufl_free_indices
            efid = expression.ufl_index_dimensions
            fi = list(zip(efi, efid))
            for pos, ind in enumerate(multiindex._indices):
                if isinstance(ind, Index):
                    fi.append((ind.count(), shape[pos]))
            fi = unique_sorted_indices(sorted(fi))
            if fi:
                fi, fid = zip(*fi)
            else:
                fi, fid = (), ()
            return Zero(shape=(), free_indices=fi, index_dimensions=fid)
        elif expression.ufl_shape == () and multiindex == ():
            return expression
        else:
            return Operator.__new__(cls)

    def __init__(self, expression, multiindex):
        # Store operands
        Operator.__init__(self, (expression, multiindex))

        # Error checking
        if not isinstance(expression, Expr):
            error("Expecting Expr instance, not %s." % ufl_err_str(expression))
        if not isinstance(multiindex, MultiIndex):
            error("Expecting MultiIndex instance, not %s." % ufl_err_str(multiindex))

        shape = expression.ufl_shape

        # Error checking
        if len(shape) != len(multiindex):
            error("Invalid number of indices (%d) for tensor "
                  "expression of rank %d:\n\t%s\n"
                  % (len(multiindex), len(expression.ufl_shape), ufl_err_str(expression)))
        if any(int(di) >= int(si) or int(di) < 0
               for si, di in zip(shape, multiindex)
               if isinstance(di, FixedIndex)):
            error("Fixed index out of range!")

        # Build tuples of free index ids and dimensions
        if 1:
            efi = expression.ufl_free_indices
            efid = expression.ufl_index_dimensions
            fi = list(zip(efi, efid))
            for pos, ind in enumerate(multiindex._indices):
                if isinstance(ind, Index):
                    fi.append((ind.count(), shape[pos]))
            fi = unique_sorted_indices(sorted(fi))
            if fi:
                fi, fid = zip(*fi)
            else:
                fi, fid = (), ()

        else:
            mfiid = [(ind.count(), shape[pos])
                     for pos, ind in enumerate(multiindex._indices)
                     if isinstance(ind, Index)]
            mfi, mfid = zip(*mfiid) if mfiid else ((), ())
            fi, fid = merge_unique_indices(expression.ufl_free_indices,
                                           expression.ufl_index_dimensions,
                                           mfi, mfid)

        # Cache free index and dimensions
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid

    ufl_shape = ()

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        A, ii = self.ufl_operands
        component = ii.evaluate(x, mapping, None, index_values)
        if derivatives:
            return A.evaluate(x, mapping, component, index_values, derivatives)
        else:
            return A.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "%s[%s]" % (parstr(self.ufl_operands[0], self),
                           self.ufl_operands[1])

    def __getitem__(self, key):
        if key == ():
            # So that one doesn't have to special case indexing of
            # expressions without shape.
            return self
        error("Attempting to index with %s, but object is already indexed: %s" % (ufl_err_str(key), ufl_err_str(self)))
