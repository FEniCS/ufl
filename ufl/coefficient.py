"""This module defines the Coefficient class and a number
of related classes, including Constant."""

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.terminal import Terminal, FormArgument
from ufl.finiteelement import FiniteElementBase, FiniteElement, VectorElement, TensorElement
from ufl.split_functions import split
from ufl.common import counted_init
from ufl.core.ufl_type import ufl_type

# --- The Coefficient class represents a coefficient in a form ---

@ufl_type()
class Coefficient(FormArgument):
    """UFL form argument type: Representation of a form coefficient."""

    # Slots are disabled here because they cause trouble in PyDOLFIN multiple inheritance pattern:
    #__slots__ = ("_count", "_element", "_repr", "_gradient", "_derivatives")
    _ufl_noslots_ = True
    _globalcount = 0

    def __init__(self, element, count=None):
        FormArgument.__init__(self)
        counted_init(self, count, Coefficient)

        ufl_assert(isinstance(element, FiniteElementBase),
            "Expecting a FiniteElementBase instance.")
        self._element = element
        self._repr = None

    def count(self):
        return self._count

    def reconstruct(self, element=None, count=None):
        # This code is shared with the FooConstant classes
        if element is None or element == self._element:
            element = self._element
        if count is None or count == self._count:
            count = self._count
        if count is self._count and element is self._element:
            return self
        ufl_assert(isinstance(element, FiniteElementBase),
                   "Expecting an element, not %s" % element)
        ufl_assert(isinstance(count, int),
                   "Expecting an int, not %s" % count)
        ufl_assert(element.value_shape() == self._element.value_shape(),
                   "Cannot reconstruct a Coefficient with a different value shape.")
        return self._reconstruct(element, count)

    def _reconstruct(self, element, count):
        # This code is class specific
        return Coefficient(element, count)

    def element(self):
        return self._element

    def shape(self):
        return self._element.value_shape()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return self._element.is_cellwise_constant()

    def domains(self):
        "Return tuple of domains related to this terminal object."
        return self._element.domains()

    def signature_data(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        count = renumbering[self]
        edata = self.element().signature_data(renumbering)
        d = self.domain()
        ddata = None if d is None else d.signature_data(renumbering)
        return ("Coefficient", count, edata, ddata)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "w_%s" % count
        else:
            return "w_{%s}" % count

    def __repr__(self):
        if self._repr is None:
            self._repr = "Coefficient(%r, %r)" % (self._element, self._count)
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Coefficient):
            return False
        if self is other:
            return True
        return (self._count == other._count and
                self._element == other._element)
    __hash__ = Terminal.__hash__

# --- Helper functions for defining constant coefficients without specifying element ---

def Constant(domain, count=None):
    """UFL value: Represents a globally constant scalar valued coefficient."""
    e = FiniteElement("Real", domain, 0)
    return Coefficient(e, count=count)

def VectorConstant(domain, dim=None, count=None):
    """UFL value: Represents a globally constant vector valued coefficient."""
    e = VectorElement("Real", domain, 0, dim)
    return Coefficient(e, count=count)

def TensorConstant(domain, shape=None, symmetry=None, count=None):
    """UFL value: Represents a globally constant tensor valued coefficient."""
    e = TensorElement("Real", domain, 0, shape=shape, symmetry=symmetry)
    return Coefficient(e, count=count)

# --- Helper functions for subfunctions on mixed elements ---

def Coefficients(element):
    """UFL value: Create a Coefficient in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(Coefficient(element))
