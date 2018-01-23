# -*- coding: utf-8 -*-
"""This module defines the class Argument and a number of related
classes (functions), including TestFunction and TrialFunction."""

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

import numbers
from ufl.utils.str import as_native_str
from ufl.utils.str import as_native_strings
from ufl.log import error
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import FormArgument
from ufl.split_functions import split
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace, FunctionSpaceProduct

# Export list for ufl.classes (TODO: not actually classes: drop? these are in ufl.*)
__all_classes__ = as_native_strings(["TestFunction", "TrialFunction", "TestFunctions", "TrialFunctions", "View"])


# --- Class representing an argument (basis function) in a form ---

@ufl_type()
class Argument(FormArgument):
    """UFL value: Representation of an argument to a form."""
    __slots__ = as_native_strings((
        "_ufl_function_space",
        "_ufl_shape",
        "_number",
        "_part",
        "_repr",
        "_initial_function_space",
    ))

    def __init__(self, function_space, number, part=None):
        FormArgument.__init__(self)

        if isinstance(function_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map the cell to
            # the default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, AbstractFunctionSpace):
            error("Expecting a FunctionSpace or FiniteElement.")

        self._ufl_function_space = function_space
        self._initial_function_space = None
        self._ufl_shape = function_space.ufl_element().value_shape()

        if not isinstance(number, numbers.Integral):
            error("Expecting an int for number, not %s" % (number,))
        if part is not None and not isinstance(part, numbers.Integral):
            error("Expecting None or an int for part, not %s" % (part,))
        self._number = number
        self._part = part

        self._repr = as_native_str("Argument(%s, %s, %s)" % (
            repr(self._ufl_function_space), repr(self._number), repr(self._part)))

    @property
    def ufl_shape(self):
        "Return the associated UFL shape."
        return self._ufl_shape

    def ufl_function_space(self):
        "Get the function space of this Argument."
        if self.is_a_view():
            return self._initial_function_space
        else:
            return self._ufl_function_space

    def ufl_domain(self):
        "Deprecated, please use .ufl_function_space().ufl_domain() instead."
        # TODO: deprecate("Argument.ufl_domain() is deprecated, please
        # use .ufl_function_space().ufl_domain() instead.")
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        "Deprecated, please use .ufl_function_space().ufl_element() instead."
        # TODO: deprecate("Argument.ufl_domain() is deprecated, please
        # use .ufl_function_space().ufl_element() instead.")
        return self._ufl_function_space.ufl_element()

    def number(self):
        "Return the Argument number."
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

    def is_a_view(self):
        return bool(self._initial_function_space != None)

    def set_view(self, function_space):
        self._initial_function_space = function_space

    def ufl_domains(self):
        "Deprecated, please use .ufl_function_space().ufl_domains() instead."
        # TODO: deprecate("Argument.ufl_domains() is deprecated,
        # please use .ufl_function_space().ufl_domains() instead.")
        return self._ufl_function_space.ufl_domains()

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
    if isinstance(function_space, FunctionSpaceProduct):
        return ArgumentProduct(function_space, 0)
    else:
        return Argument(function_space, 0, part)


def TrialFunction(function_space, part=None):
    """UFL value: Create a trial function argument to a form."""
    if isinstance(function_space, FunctionSpaceProduct):
        return ArgumentProduct(function_space, 1)
    else:
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

def ArgumentProduct(function_space, number):
    if not isinstance(function_space, FunctionSpaceProduct):
        error("ArgumentProduct should be used with FunctionSpaceProduct")

    subspaces = function_space.sub_spaces()
    arguments = list()
    i=0
    # Build list of Argument objects with _part=<subspace index>
    for s in subspaces:
        arguments.append(Argument(s, number, i))
        i = i+1
    return tuple(arguments)

## New function to define the view of an argument
def View(argument, function_space):
    assert isinstance(function_space, FunctionSpace)
    assert isinstance(argument, Argument)
    argument_view = Argument(function_space, argument.number(), argument.part())
    argument_view.set_view(argument.function_space())
    return argument_view;

