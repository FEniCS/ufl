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

import numbers
from ufl.log import error
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import FormArgument
from ufl.split_functions import split
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.form import BaseForm
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace, MixedFunctionSpace
from ufl.duals import is_primal, is_dual

# Export list for ufl.classes (TODO: not actually classes: drop? these are in ufl.*)
__all_classes__ = ["TestFunction", "TrialFunction", "TestFunctions", "TrialFunctions"]


# --- Class representing an argument (basis function) in a form ---

class BaseArgument(object):
    """UFL value: Representation of an argument to a form."""
    __slots__ = ()
    _ufl_is_abstract_ = True

    def __getnewargs__(self):
        return (self._ufl_function_space, self._number, self._part)

    def __init__(self, function_space, number, part=None):

        if isinstance(function_space, FiniteElementBase):
            # For legacy support for UFL files using cells, we map the cell to
            # the default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, AbstractFunctionSpace):
            error("Expecting a FunctionSpace or FiniteElement.")

        self._ufl_function_space = function_space
        self._ufl_shape = function_space.ufl_element().value_shape()

        if not isinstance(number, numbers.Integral):
            error("Expecting an int for number, not %s" % (number,))
        if part is not None and not isinstance(part, numbers.Integral):
            error("Expecting None or an int for part, not %s" % (part,))
        self._number = number
        self._part = part

        self._repr = "BaseArgument(%s, %s, %s)" % (
            repr(self._ufl_function_space), repr(self._number), repr(self._part))

    @property
    def ufl_shape(self):
        "Return the associated UFL shape."
        return self._ufl_shape

    def ufl_function_space(self):
        "Get the function space of this Argument."
        return self._ufl_function_space

    def ufl_domain(self):
        """Return the UFL domain."""
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        """Return The UFL element."""
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

    def ufl_domains(self):
        """Return UFL domains."""
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


@ufl_type()
class Argument(FormArgument, BaseArgument):
    """UFL value: Representation of an argument to a form."""
    __slots__ = (
        "_ufl_function_space",
        "_ufl_shape",
        "_number",
        "_part",
        "_repr",
    )

    _primal = True
    _dual = False

    __getnewargs__ = BaseArgument.__getnewargs__
    __str__ = BaseArgument.__str__
    _ufl_signature_data_ = BaseArgument._ufl_signature_data_
    __eq__ = BaseArgument.__eq__

    def __new__(cls, *args, **kw):
        if args[0] and is_dual(args[0]):
            return Coargument(*args, **kw)
        return super().__new__(cls)

    def __init__(self, function_space, number, part=None):
        FormArgument.__init__(self)
        BaseArgument.__init__(self, function_space, number, part)

        self._repr = "Argument(%s, %s, %s)" % (
            repr(self._ufl_function_space), repr(self._number), repr(self._part))

    def ufl_domains(self):
        return BaseArgument.ufl_domains(self)

    def __repr__(self):
        return self._repr


@ufl_type()
class Coargument(BaseForm, BaseArgument):
    """UFL value: Representation of an argument to a form in a dual space."""
    __slots__ = (
        "_ufl_function_space",
        "_ufl_shape",
        "_arguments",
        "ufl_operands",
        "_number",
        "_part",
        "_repr",
        "_hash"
    )

    _primal = False
    _dual = True

    def __new__(cls, *args, **kw):
        if args[0] and is_primal(args[0]):
            raise ValueError('ufl.Coargument takes in a dual space! If you want to define an argument in the primal space you should use ufl.Argument.')
        return super().__new__(cls)

    def __init__(self, function_space, number, part=None):
        BaseArgument.__init__(self, function_space, number, part)
        BaseForm.__init__(self)

        self.ufl_operands = ()
        self._hash = None
        self._repr = "Coargument(%s, %s, %s)" % (
            repr(self._ufl_function_space), repr(self._number), repr(self._part))

    def _analyze_form_arguments(self):
        "Analyze which Argument and Coefficient objects can be found in the form."
        # Define canonical numbering of arguments and coefficients
        self._arguments = (Argument(self._ufl_function_space, 0),)

    def equals(self, other):
        if type(other) is not Coargument:
            return False
        if self is other:
            return True
        return (self._ufl_function_space == other._ufl_function_space and
                self._number == other._number and self._part == other._part)

    def __hash__(self):
        """Hash code for use in dicts."""
        return hash(("Coargument",
                     hash(self._ufl_function_space),
                     self._number,
                     self._part))

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
    if isinstance(function_space, MixedFunctionSpace):
        return [Argument(function_space.ufl_sub_space(i), number, i)
                for i in range(function_space.num_sub_spaces())]
    else:
        return split(Argument(function_space, number))


def TestFunctions(function_space):
    """UFL value: Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return Arguments(function_space, 0)


def TrialFunctions(function_space):
    """UFL value: Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    return Arguments(function_space, 1)
