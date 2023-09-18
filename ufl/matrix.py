"""This module defines the Matrix class."""
# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022.

from ufl.form import BaseForm
from ufl.core.ufl_type import ufl_type
from ufl.argument import Argument
from ufl.functionspace import AbstractFunctionSpace
from ufl.utils.counted import Counted


# --- The Matrix class represents a matrix, an assembled two form ---

@ufl_type()
class Matrix(BaseForm, Counted):
    """An assemble linear operator between two function spaces."""

    __slots__ = (
        "_count",
        "_counted_class",
        "_ufl_function_spaces",
        "ufl_operands",
        "_repr",
        "_hash",
        "_ufl_shape",
        "_arguments",
        "_coefficients",
        "_domains")

    def __getnewargs__(self):
        """Get new args."""
        return (self._ufl_function_spaces[0], self._ufl_function_spaces[1],
                self._count)

    def __init__(self, row_space, column_space, count=None):
        """Initialise."""
        BaseForm.__init__(self)
        Counted.__init__(self, count, Matrix)

        if not isinstance(row_space, AbstractFunctionSpace):
            raise ValueError("Expecting a FunctionSpace as the row space.")

        if not isinstance(column_space, AbstractFunctionSpace):
            raise ValueError("Expecting a FunctionSpace as the column space.")

        self._ufl_function_spaces = (row_space, column_space)

        self.ufl_operands = ()
        self._domains = None
        self._hash = None
        self._repr = f"Matrix({self._ufl_function_spaces[0]!r}, {self._ufl_function_spaces[1]!r}, {self._count!r})"

    def ufl_function_spaces(self):
        """Get the tuple of function spaces of this coefficient."""
        return self._ufl_function_spaces

    def _analyze_form_arguments(self):
        """Define arguments of a matrix when considered as a form."""
        self._arguments = (Argument(self._ufl_function_spaces[0], 0),
                           Argument(self._ufl_function_spaces[1], 1))
        self._coefficients = ()

    def _analyze_domains(self):
        """Analyze which domains can be found in a Matrix."""
        from ufl.domain import join_domains
        # Collect unique domains
        self._domains = join_domains([fs.ufl_domain() for fs in self._ufl_function_spaces])

    def __str__(self):
        """Format as a string."""
        count = str(self._count)
        if len(count) == 1:
            return f"A_{count}"
        else:
            return f"A_{{{count}}}"

    def __repr__(self):
        """Representation."""
        return self._repr

    def __hash__(self):
        """Hash."""
        if self._hash is None:
            self._hash = hash(self._repr)
        return self._hash

    def equals(self, other):
        """Check equality."""
        if type(other) is not Matrix:
            return False
        if self is other:
            return True
        return self._count == other._count and self._ufl_function_spaces == other._ufl_function_spaces
