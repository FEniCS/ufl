# -*- coding: utf-8 -*-
"""This module defines the Matrix class."""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from ufl.log import error
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import FormArgument
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace
from ufl.utils.counted import counted_init


# --- The Matrix class represents a matrix, an assembled two form ---


@ufl_type()
class Matrix(FormArgument):
    """UFL form argument type: Parent Representation of a form coefficient."""

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    # __slots__ = ("_count", "_ufl_function_space", "_repr", "_ufl_shape")
    _ufl_noslots_ = True
    _globalcount = 0

    def __getnewargs__(self):
        return (self._ufl_function_space[0], self._ufl_function_space[1], self._count)

    def __init__(self, row_space, column_space, count=None):
        FormArgument.__init__(self)
        counted_init(self, count, Matrix)

        if isinstance(row_space, FiniteElementBase) and isinstance(column_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map
            # the cell to The Default Mesh
            element = row_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, AbstractFunctionSpace):
            error("Expecting a FunctionSpace or FiniteElement.")

        self._ufl_function_spaces = (row_space, column_space)

        self._repr = "Matrix(%s,%s, %s)" % (
            repr(self._ufl_function_spaces[0]), repr(self._ufl_function_spaces[1]), repr(self._count))

    def count(self):
        return self._count

    def ufl_function_spaces(self):
        "Get the tuple of function spaces of this coefficient."
        return self._ufl_function_spaces

    def ufl_row_space(self):
        return self._ufl_function_spaces[0]

    def ufl_column_space(self):
        return self._ufl_function_spaces[1]

    def _ufl_signature_data_(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        count = renumbering[self]
        row_fsdata = self._ufl_function_spaces[0]._ufl_signature_data_(renumbering)
        col_fsdata = self._ufl_function_spaces[1]._ufl_signature_data_(renumbering)
        return ("Matrix", count, row_fsdata, col_fsdata)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "A_%s" % count
        else:
            return "A_{%s}" % count

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if self is other:
            return True
        return (self._count == other._count and
                self._ufl_function_spaces == other._ufl_function_spaces)
