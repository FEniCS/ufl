# -*- coding: utf-8 -*-
"""This module defines the Coefficient class and a number
of related classes, including Constant."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008-2009.
# Modified by Massimiliano Leoni, 2016.

from ufl.utils.str import as_native_str
from ufl.log import error
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import FormArgument
from ufl.finiteelement import FiniteElementBase, FiniteElement, VectorElement, TensorElement
from ufl.domain import as_domain, default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace
from ufl.split_functions import split
from ufl.utils.counted import counted_init

# --- The Coefficient class represents a coefficient in a form ---


@ufl_type()
class Coefficient(FormArgument):
    """UFL form argument type: Representation of a form coefficient."""

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    # __slots__ = ("_count", "_ufl_function_space", "_repr", "_ufl_shape")
    _ufl_noslots_ = True
    _globalcount = 0

    def __init__(self, function_space, count=None):
        FormArgument.__init__(self)
        counted_init(self, count, Coefficient)

        if isinstance(function_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map
            # the cell to The Default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, AbstractFunctionSpace):
            error("Expecting a FunctionSpace or FiniteElement.")

        self._ufl_function_space = function_space
        self._ufl_shape = function_space.ufl_element().value_shape()

        self._repr = as_native_str("Coefficient(%s, %s)" % (
            repr(self._ufl_function_space), repr(self._count)))

    def count(self):
        return self._count

    @property
    def ufl_shape(self):
        "Return the associated UFL shape."
        return self._ufl_shape

    def ufl_function_space(self):
        "Get the function space of this coefficient."
        return self._ufl_function_space

    def ufl_domain(self):
        "Shortcut to get the domain of the function space of this coefficient."
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        "Shortcut to get the finite element of the function space of this coefficient."
        return self._ufl_function_space.ufl_element()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return self.ufl_element().is_cellwise_constant()

    def ufl_domains(self):
        "Return tuple of domains related to this terminal object."
        return self._ufl_function_space.ufl_domains()

    def _ufl_signature_data_(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        count = renumbering[self]
        fsdata = self._ufl_function_space._ufl_signature_data_(renumbering)
        return ("Coefficient", count, fsdata)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "w_%s" % count
        else:
            return "w_{%s}" % count

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Coefficient):
            return False
        if self is other:
            return True
        return (self._count == other._count and
                self._ufl_function_space == other._ufl_function_space)


# --- Subclasses for defining constant coefficients without
# --- specifying element ---

class Constant(ufl.Coefficient):
    """UFL value: Represents a globally constant scalar valued coefficient."""
    def __init__(self, domain, count=None):
        domain = as_domain(domain)
        element = FiniteElement("Real", domain.ufl_cell(), 0)
        fs = FunctionSpace(domain, element)
        super().__init__(fs, count=count)


class VectorConstant(ufl.Coefficient):
    """UFL value: Represents a globally constant vector valued coefficient."""
    def __init__(self, domain, dim=None, count=None):
        domain = as_domain(domain)
        element = VectorElement("Real", domain.ufl_cell(), 0, dim)
        fs = FunctionSpace(domain, element)
        super().__init__(fs, count=count)


class TensorConstant(ufl.Coefficient):
    """UFL value: Represents a globally constant tensor valued coefficient."""
    def __init__(self, domain, shape=None, symmetry=None, count=None):
        domain = as_domain(domain)
        element = TensorElement("Real", domain.ufl_cell(), 0, shape=shape,
                                symmetry=symmetry)
        fs = FunctionSpace(domain, element)
        super().__init__(fs, count=count)


# --- Helper functions for subfunctions on mixed elements ---

def Coefficients(function_space):
    """UFL value: Create a Coefficient in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(Coefficient(function_space))
