# -*- coding: utf-8 -*-
"""This module defines the class Argument and a number of related
classes (functions), including TestFunction and TrialFunction."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009.
# Modified by Massimiliano Leoni, 2016.
# Modified by Cecile Daversin-Catty, 2018.
# Modified by Jørgen S. Dokken, 2022.

import numbers
import typing

from ufl.core.terminal import FormArgument
from ufl.core.ufl_type import ufl_type
from ufl.finiteelement import FiniteElementBase
from ufl.functionspace import AbstractFunctionSpace, MixedFunctionSpace
from ufl.log import error
from ufl.split_functions import split

# Export list for ufl.classes (TODO: not actually classes: drop? these are in ufl.*)
__all_classes__ = ["TestFunction", "TrialFunction", "TestFunctions", "TrialFunctions"]


# --- Class representing an argument (basis function) in a form ---

@ufl_type()
class Argument(FormArgument):
    """UFL value: Representation of an argument to a form."""
    __slots__ = (
        "_ufl_function_space",
        "_ufl_shape",
        "_number",
        "_part",
        "_repr",
    )

    def __init__(self, function_space: typing.Union[FiniteElementBase, AbstractFunctionSpace],
                 number: numbers.Integral, part: numbers.Integral = None):
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
        self._ufl_shape = function_space.ufl_element().value_shape()

        if not isinstance(number, numbers.Integral):
            error(f"Expecting an int for number, not {number}")
        if part is not None and not isinstance(part, numbers.Integral):
            error(f"Expecting None or an int for part, not {part}")
        self._number = number
        self._part = part

        self._repr = "Argument({0:s}, {1:s}, {2:s})".format(
            repr(self._ufl_function_space), repr(self._number), repr(self._part))

    @property
    def ufl_shape(self):
        """Return the associated UFL shape."""
        return self._ufl_shape

    def ufl_function_space(self):
        """Get the function space of this Argument."""
        return self._ufl_function_space

    def number(self):
        """Return the Argument number."""
        return self._number

    def part(self):
        return self._part

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # TODO: Should in principle do like with Coefficient,
        # but that may currently simplify away some arguments
        # we want to keep, or? See issue#13.
        # When we can annotate zero with arguments, we can change this.
        return False

    def _ufl_signature_data_(self, renumbering):
        """Signature data for form arguments depend on the global numbering of the form arguments and domains."""
        fsdata = self._ufl_function_space._ufl_signature_data_(renumbering)
        return ("Argument", self._number, self._part, fsdata)

    def __str__(self):
        number = str(self._number)
        s = ""
        # Add curly brackets around sub-indices if bigger than 9
        number = number if len(number) == 1 else f"{{{number}}}"
        s += f"v_{number}"
        if self._part is not None:
            part = str(self._part)
            # Add curly brackets around sup-indices if bigger than 9
            part = part if len(part) == 1 else f"{{{part}}}"
            s += f"^{part}"
        return s

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        """
        Deliberately comparing exact type and not using isinstance here,
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


def TestFunction(function_space: typing.Union[FiniteElementBase, AbstractFunctionSpace],
                 part: numbers.Integral = None):
    """UFL value: Create a test function argument to a form."""
    return Argument(function_space, 0, part)


def TrialFunction(function_space: typing.Union[FiniteElementBase, AbstractFunctionSpace], part: numbers.Integral = None):
    """UFL value: Create a trial function argument to a form."""
    return Argument(function_space, 1, part)


# --- Helper functions for creating subfunctions on mixed elements ---

def Arguments(function_space: typing.Union[FiniteElementBase, AbstractFunctionSpace], number: numbers.Integral):
    """
    UFL value: Create an Argument in a mixed space, and return a
    tuple with the function components corresponding to the subelements.
    """
    if isinstance(function_space, MixedFunctionSpace):
        return [Argument(function_space.ufl_sub_space(i), number, i)
                for i in range(function_space.num_sub_spaces())]
    else:
        return split(Argument(function_space, number))


def TestFunctions(function_space: typing.Union[FiniteElementBase, AbstractFunctionSpace]):
    """
    UFL value: Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements.
    """
    return Arguments(function_space, 0)


def TrialFunctions(function_space: typing.Union[FiniteElementBase, AbstractFunctionSpace]):
    """
    UFL value: Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements.
    """
    return Arguments(function_space, 1)
