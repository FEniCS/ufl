# -*- coding: utf-8 -*-
"""This module defines special types for representing mapping of expressions to expressions."""

# Copyright (C) 2014 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.log import error
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.coefficient import Cofunction
from ufl.argument import Coargument


# --- Non-tensor types ---

@ufl_type(num_ops="varying")
class ExprList(Operator):
    "List of Expr objects. For internal use, never to be created by end users."
    __slots__ = ()

    def __init__(self, *operands):
        Operator.__init__(self, operands)
        # Enable Cofunction/Coargument for BaseForm differentiation
        if not all(isinstance(i, (Expr, Cofunction, Coargument)) for i in operands):
            error("Expecting Expr, Cofunction or Coargument in ExprList.")

    def __getitem__(self, i):
        return self.ufl_operands[i]

    def __len__(self):
        return len(self.ufl_operands)

    def __iter__(self):
        return iter(self.ufl_operands)

    def __str__(self):
        return "ExprList(*(%s,))" % ", ".join(str(i) for i in self.ufl_operands)

    def __repr__(self):
        r = "ExprList(*%s)" % repr(self.ufl_operands)
        return r

    @property
    def ufl_shape(self):
        error("A non-tensor type has no ufl_shape.")

    @property
    def ufl_free_indices(self):
        error("A non-tensor type has no ufl_free_indices.")

    def free_indices(self):
        error("A non-tensor type has no free_indices.")

    @property
    def ufl_index_dimensions(self):
        error("A non-tensor type has no ufl_index_dimensions.")

    def index_dimensions(self):
        error("A non-tensor type has no index_dimensions.")


@ufl_type(num_ops="varying")
class ExprMapping(Operator):
    "Mapping of Expr objects. For internal use, never to be created by end users."
    __slots__ = ()

    def __init__(self, *operands):
        Operator.__init__(self, operands)
        if not all(isinstance(e, Expr) for e in operands):
            error("Expecting Expr in ExprMapping.")

    def ufl_domains(self):
        # Because this type can act like a terminal if it has no
        # operands, we need to override some recursive operations
        if self.ufl_operands:
            return Operator.ufl_domains()
        else:
            return []

    # def __getitem__(self, key):
    #     return self.ufl_operands[key]

    # def __len__(self):
    #     return len(self.ufl_operands) // 2

    # def __iter__(self):
    #     return iter(self.ufl_operands[::2])

    def __str__(self):
        return "ExprMapping(*%s)" % repr(self.ufl_operands)

    def __repr__(self):
        r = "ExprMapping(*%s)" % repr(self.ufl_operands)
        return r

    @property
    def ufl_shape(self):
        error("A non-tensor type has no ufl_shape.")

    @property
    def ufl_free_indices(self):
        error("A non-tensor type has no ufl_free_indices.")

    def free_indices(self):
        error("A non-tensor type has no free_indices.")

    @property
    def ufl_index_dimensions(self):
        error("A non-tensor type has no ufl_index_dimensions.")

    def index_dimensions(self):
        error("A non-tensor type has no index_dimensions.")
