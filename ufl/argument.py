"""This module defines the class Argument and a number of related
classes (functions), including TestFunction and TrialFunction."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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

from ufl.assertions import ufl_assert
from ufl.common import Counted, product
from ufl.terminal import FormArgument
from ufl.split_functions import split
from ufl.finiteelement import FiniteElementBase

# --- Class representing an argument (basis function) in a form ---

class Argument(FormArgument, Counted):
    """UFL value: Representation of an argument to a form."""
    __slots__ = ("_repr", "_element",)
    _globalcount = 0

    def __init__(self, element, count=None):
        FormArgument.__init__(self)
        Counted.__init__(self, count, Argument)
        ufl_assert(isinstance(element, FiniteElementBase),
            "Expecting a FiniteElementBase instance.")
        self._element = element
        self._repr = "Argument(%r, %r)" % (self._element, self._count)

    def reconstruct(self, element=None, count=None):
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
                   "Cannot reconstruct an Argument with a different value shape.")
        return Argument(element, count)

    def element(self):
        return self._element

    def shape(self):
        return self._element.value_shape()

    def cell(self):
        return self._element.cell()

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "v_%s" % count
        else:
            return "v_{%s}" % count

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return isinstance(other, Argument) and self._element == other._element and self._count == other._count

# --- Helper functions for pretty syntax ---

def TestFunction(element):
    """UFL value: Create a test function argument to a form."""
    return Argument(element, -2)

def TrialFunction(element):
    """UFL value: Create a trial function argument to a form."""
    return Argument(element, -1)

# --- Helper functions for creating subfunctions on mixed elements ---

def Arguments(element):
    """UFL value: Create an Argument in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(Argument(element))

def TestFunctions(element):
    """UFL value: Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(TestFunction(element))

def TrialFunctions(element):
    """UFL value: Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(TrialFunction(element))
