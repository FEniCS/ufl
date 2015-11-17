# -*- coding: utf-8 -*-
"""This module defines the class Argument and a number of related
classes (functions), including TestFunction and TrialFunction."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal, FormArgument
from ufl.split_functions import split
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace


# Export list for ufl.classes (TODO: not actually classes: drop? these are in ufl.*)
__all_classes__ = ["TestFunction", "TrialFunction", "TestFunctions", "TrialFunctions"]


# --- Class representing an argument (basis function) in a form ---

@ufl_type()
class Argument(FormArgument):
    """UFL value: Representation of an argument to a form."""
    __slots__ = ("_ufl_function_space", "_ufl_shape", "_number", "_part", "_repr")

    def __init__(self, function_space, number, part=None):
        FormArgument.__init__(self)

        if isinstance(function_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map the cell to The Default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, AbstractFunctionSpace):
            error("Expecting a FunctionSpace or FiniteElement.")

        self._ufl_function_space = function_space
        self._ufl_shape = function_space.ufl_element().value_shape()

        ufl_assert(isinstance(number, int),
                   "Expecting an int for number, not %s" % (number,))
        ufl_assert(part is None or isinstance(part, int),
                   "Expecting None or an int for part, not %s" % (part,))
        self._number = number
        self._part = part

        self._repr = "Argument(%r, %r, %r)" % (self._ufl_function_space, self._number, self._part)

    @property
    def ufl_shape(self):
        return self._ufl_shape

    def ufl_function_space(self):
        "Get the function space of this Argument."
        return self._ufl_function_space

    def ufl_domain(self):
        #TODO: deprecate("Argument.ufl_domain() is deprecated, please use .ufl_function_space().ufl_domain() instead.")
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        #TODO: deprecate("Argument.ufl_domain() is deprecated, please use .ufl_function_space().ufl_element() instead.")
        return self._ufl_function_space.ufl_element()

    def element(self):
        deprecate("Argument.element() is deprecated, please use Argument.ufl_element() instead.")
        return self.ufl_element()

    def number(self):
        return self._number

    def part(self):
        return self._part

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # TODO: Should in principle do like with Coefficient,
        # but that may currently simplify away some arguments
        # we want to keep, or? See issue#13.
        # When we can annotate zero with arguments, we can change this.
        return False

    def ufl_domains(self):
        "Return tuple of domains related to this terminal object."
        d = self.ufl_domain() # TODO: Can we get more than one domain from a mixed function space?
        if d is None:
            return ()
        else:
            return (d,)

    def _ufl_signature_data_(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        fsdata = self._ufl_function_space._ufl_signature_data_(renumbering)
        return ("Argument", self._number, self._part, fsdata)

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
                self._ufl_function_space == other._ufl_function_space)

# --- Helper functions for pretty syntax ---

def TestFunction(function_space, part=None):
    """UFL value: Create a test function argument to a form."""
    return Argument(function_space, 0, part)

def TrialFunction(function_space, part=None):
    """UFL value: Create a trial function argument to a form."""
    return Argument(function_space, 1, part)

# --- Helper functions for creating subfunctions on mixed elements ---

def Arguments(function_space, number):
    """UFL value: Create an Argument in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return split(Argument(function_space, number))

def TestFunctions(function_space):
    """UFL value: Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return Arguments(function_space, 0)

def TrialFunctions(function_space):
    """UFL value: Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return Arguments(function_space, 1)
