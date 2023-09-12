# -*- coding: utf-8 -*-

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008
# Modified by Massimiliano Leoni, 2016

from ufl.core.expr import Expr
from ufl.core.ufl_type import ufl_type


@ufl_type(is_abstract=True, is_terminal=False)
class Operator(Expr):
    "Base class for all operators, i.e. non-terminal expression types."
    __slots__ = ("ufl_operands",)

    def __init__(self, operands=None):
        Expr.__init__(self)

        # If operands is None, the type sets this itself. This is to
        # get around some tricky too-fancy __new__/__init__ design in
        # algebra.py, for now.  It would be nicer to make the classes
        # in algebra.py pass operands here.
        if operands is not None:
            self.ufl_operands = operands

    def _ufl_expr_reconstruct_(self, *operands):
        "Return a new object of the same type with new operands."
        return self._ufl_class_(*operands)

    def _ufl_signature_data_(self):
        return self._ufl_typecode_

    def _ufl_compute_hash_(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        return hash((self._ufl_typecode_,) + tuple(hash(o) for o in self.ufl_operands))

    def __repr__(self):
        "Default repr string construction for operators."
        # This should work for most cases
        r = "%s(%s)" % (self._ufl_class_.__name__,
                        ", ".join(repr(op) for op in self.ufl_operands))
        return r
