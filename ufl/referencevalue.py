# -*- coding: utf-8 -*-
"Representation of the reference value of a function."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.ufl_type import ufl_type
from ufl.core.operator import Operator
from ufl.core.terminal import FormArgument


@ufl_type(num_ops=1,
          is_index_free=True,
          is_terminal_modifier=True,
          is_in_reference_frame=True)
class ReferenceValue(Operator):
    "Representation of the reference cell value of a form argument."
    __slots__ = ()

    def __init__(self, f):
        if not isinstance(f, FormArgument):
            raise ValueError("Can only take reference value of form arguments.")
        Operator.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_element().reference_value_shape

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        raise NotImplementedError()

    def __str__(self):
        return f"reference_value({self.ufl_operands[0]})"
