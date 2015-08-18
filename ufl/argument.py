# -*- coding: utf-8 -*-
"""This module defines the class Argument and a number of related
classes (functions), including TestFunction and TrialFunction."""

# Copyright (C) 2008-2014 Martin Sandve Alnæs
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

from ufl.log import deprecate
from ufl.assertions import ufl_assert
from ufl.core.terminal import Terminal, FormArgument
from ufl.split_functions import split
from ufl.finiteelement import FiniteElementBase
from ufl.core.ufl_type import ufl_type

# --- Class representing an argument (basis function) in a form ---

@ufl_type()
class Argument(FormArgument):
    """UFL value: Representation of an argument to a form."""
    __slots__ = ("_element", "_number", "_part", "_repr")

    def __init__(self, element, number, part=None):
        FormArgument.__init__(self)
        ufl_assert(isinstance(element, FiniteElementBase),
                   "Expecting an element, not %s" % (element,))
        ufl_assert(isinstance(number, int),
                   "Expecting an int for number, not %s" % (number,))
        ufl_assert(part is None or isinstance(part, int),
                   "Expecting None or an int for part, not %s" % (part,))
        self._element = element
        self._number = number
        self._part = part
        self._repr = "Argument(%r, %r, %r)" % (self._element, self._number, self._part)

    def reconstruct(self, element=None, number=None, part=None):
        if element is None or (element == self._element): # TODO: Is the == here a workaround for some bug?
            element = self._element
        if number is None:
            number = self._number
        if part is None:
            part = self._part
        if number == self._number and part == self._part and element is self._element:
            return self
        ufl_assert(element.value_shape() == self._element.value_shape(),
                   "Cannot reconstruct an Argument with a different value shape.")
        return Argument(element, number, part)

    def element(self):
        return self._element

    def number(self):
        return self._number

    def part(self):
        return self._part

    def count(self):
        deprecate("The count of an Argument has been replaced with number() and part().")
        ufl_assert(self.part() is None, "Deprecation transition for count() will not work with parts.")
        return self.number() # I think this will work ok in most cases during the deprecation transition

    @property
    def ufl_shape(self):
        return self._element.value_shape()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # TODO: Should in principle do like with Coefficient,
        # but that may currently simplify away some arguments
        # we want to keep, or? See issue#13.
        # When we can annotate zero with arguments, we can change this.
        return False

    def domains(self):
        "Return tuple of domains related to this terminal object."
        return self._element.domains()

    def signature_data(self, domain_numbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        s = self._element.signature_data(domain_numbering=domain_numbering)
        return ("Argument", self._number, self._part) + s

    def signature_data(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        edata = self.element().signature_data(renumbering)
        d = self.domain()
        ddata = None if d is None else d.signature_data(renumbering)
        return ("Coefficient", self._number, self._part, edata, ddata)

    def __str__(self):
        number = str(self._number)
        if len(number) == 1:
            s = "v_%s" % number
        else:
            s = "v_{%s}" % number
        if self._part is not None:
            part = str(self._part)
            if len(part) == 1:
                s = "%s^%s" % (s, part)
            else:
                s = "%s^{%s}" % (s, part)
        return s

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        """Deliberately comparing exact type and not using isinstance here,
        meaning eventual subclasses must reimplement this function to work
        correctly, and instances of this class will compare not equal to
        instances of eventual subclasses. The overloading allows
        subclasses to distinguish between test and trial functions
        with a different non-ufl payload, such as dolfin FunctionSpace
        with different mesh. This is necessary because arguments with the
        same element and argument number are always equal from a pure ufl
        point of view, e.g. TestFunction(V1) == TestFunction(V2) if V1 and V2
        are the same ufl element but different dolfin function spaces.
        """
        return (type(self) == type(other) and
                self._number == other._number and
                self._part == other._part and
                self._element == other._element)

# --- Helper functions for pretty syntax ---

def TestFunction(element, part=None):
    """UFL value: Create a test function argument to a form."""
    return Argument(element, 0, part)

def TrialFunction(element, part=None):
    """UFL value: Create a trial function argument to a form."""
    return Argument(element, 1, part)

# --- Helper functions for creating subfunctions on mixed elements ---

def Arguments(element, number):
    """UFL value: Create an Argument in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(Argument(element, number))

def TestFunctions(element):
    """UFL value: Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return Arguments(element, 0)

def TrialFunctions(element):
    """UFL value: Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return Arguments(element, 1)
