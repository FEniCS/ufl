# -*- coding: utf-8 -*-
"""This module defines the Masked class."""

# Copyright (C) 2020 Imperial College London and others
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Koki Sagiyama 2020

from ufl.constantvalue import Zero
from ufl.core.ufl_type import ufl_type
from ufl.core.operator import Operator
from ufl.precedence import parstr


@ufl_type(num_ops=2, is_terminal_modifier=True, inherit_shape_from_operand=0, inherit_indices_from_operand=0)
class Masked(Operator):
    __slots__ = (
        "ufl_shape",
        "ufl_free_indices",
        "ufl_index_dimensions",
    )

    def __new__(cls, expression, transform_op):
        if isinstance(expression, Zero):
            # Zero-simplify indexed Zero objects
            shape = expression.ufl_shape
            fi = expression.ufl_free_indices
            fid = expression.ufl_index_dimensions
            return Zero(shape=shape, free_indices=fi, index_dimensions=fid)
        else:
            return Operator.__new__(cls)

    def __init__(self, expression, transform_op):
        # Store operands
        Operator.__init__(self, (expression, transform_op))

    def ufl_element(self):
        "Shortcut to get the finite element of the function space of the operand."
        return self.ufl_operands[0].ufl_element()

    def __str__(self):
        return "%s[%s]" % (parstr(self.ufl_operands[0], self),
                           self.ufl_operands[1])
