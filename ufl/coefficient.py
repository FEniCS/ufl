"""This module defines the Coefficient class and a number
of related classes, including Constant."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
#
# First added:  2008-03-14
# Last changed: 2011-10-20

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.terminal import FormArgument
from ufl.finiteelement import FiniteElementBase, FiniteElement, VectorElement, TensorElement
from ufl.split_functions import split

# --- The Coefficient class represents a coefficient in a form ---

class Coefficient(FormArgument):
    """UFL form argument type: Representation of a form coefficient."""

    # Slots are disabled here because they cause trouble in PyDOLFIN multiple inheritance pattern:
    #__slots__ = ("_count", "_element", "_repr", "_gradient", "_derivatives")
    _globalcount = 0

    def __init__(self, element, count=None):
        FormArgument.__init__(self, count, Coefficient)
        ufl_assert(isinstance(element, FiniteElementBase),
            "Expecting a FiniteElementBase instance.")
        self._element = element
        self._repr = None

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

    def cell(self):
        return self._element.cell()

    def domain(self):
        return self._element.domain()

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

# --- Subclasses for defining constant coefficients without specifying element ---

class ConstantBase(Coefficient):
    __slots__ = ()
    def __init__(self, element, count):
        Coefficient.__init__(self, element, count)

class Constant(ConstantBase):
    """UFL value: Represents a globally constant scalar valued coefficient."""
    __slots__ = ()

    def __init__(self, domain, count=None):
        e = FiniteElement("Real", domain, 0)
        ConstantBase.__init__(self, e, count)
        self._repr = "Constant(%r, %r)" % (e.domain(), self._count)

    def _reconstruct(self, element, count):
        return Constant(element.domain(), count)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "c_%s" % count
        else:
            return "c_{%s}" % count

class VectorConstant(ConstantBase):
    """UFL value: Represents a globally constant vector valued coefficient."""
    __slots__ = ()

    def __init__(self, domain, dim=None, count=None):
        e = VectorElement("Real", domain, 0, dim)
        ConstantBase.__init__(self, e, count)
        ufl_assert(self._repr is None, "Repr should not have been set yet!")
        self._repr = "VectorConstant(%r, %r, %r)" % (e.domain(), e.value_shape()[0], self._count)

    def _reconstruct(self, element, count):
        return VectorConstant(element.domain(), element.value_shape()[0], count)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "C_%s" % count
        else:
            return "C_{%s}" % count

class TensorConstant(ConstantBase):
    """UFL value: Represents a globally constant tensor valued coefficient."""
    __slots__ = ()

    def __init__(self, domain, shape=None, symmetry=None, count=None):
        e = TensorElement("Real", domain, 0, shape=shape, symmetry=symmetry)
        ConstantBase.__init__(self, e, count)
        ufl_assert(self._repr is None, "Repr should not have been set yet!")
        self._repr = "TensorConstant(%r, %r, %r, %r)" % (e.domain(), e.value_shape(), e._symmetry, self._count)

    def _reconstruct(self, element, count):
        e = element
        return TensorConstant(e.domain(), e.value_shape(), e._symmetry, count)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "C_%s" % count
        else:
            return "C_{%s}" % count

# --- Helper functions for subfunctions on mixed elements ---

def Coefficients(element):
    """UFL value: Create a Coefficient in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(Coefficient(element))
