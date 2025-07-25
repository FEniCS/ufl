"""Argument.

This module defines the class Argument and a number of related
classes (functions), including TestFunction and TrialFunction.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009.
# Modified by Massimiliano Leoni, 2016.
# Modified by Cecile Daversin-Catty, 2018.
# Modified by Ignacia Fierro-Piccardo 2023.

import numbers

from ufl.core.terminal import FormArgument
from ufl.core.ufl_type import ufl_type
from ufl.duals import is_dual, is_primal
from ufl.form import BaseForm
from ufl.functionspace import AbstractFunctionSpace, MixedFunctionSpace
from ufl.split_functions import split

# Export list for ufl.classes (TODO: not actually classes: drop? these are in ufl.*)
__all_classes__ = ["TestFunction", "TrialFunction", "TestFunctions", "TrialFunctions"]


# --- Class representing an argument (basis function) in a form ---


class BaseArgument:
    """UFL value: Representation of an argument to a form."""

    __slots__ = ()
    _ufl_is_abstract_ = True

    def __getnewargs__(self):
        """Get new args."""
        return (self._ufl_function_space, self._number, self._part)

    def __init__(self, function_space, number, part=None):
        """Initialise."""
        if not isinstance(function_space, AbstractFunctionSpace):
            raise ValueError("Expecting a FunctionSpace.")

        self._ufl_function_space = function_space
        self._ufl_shape = function_space.value_shape

        if not isinstance(number, numbers.Integral):
            raise ValueError(f"Expecting an int for number, not {number}")
        if part is not None and not isinstance(part, numbers.Integral):
            raise ValueError(f"Expecting None or an int for part, not {part}")
        self._number = number
        self._part = part

        self._repr = f"BaseArgument({self._ufl_function_space}, {self._number}, {self._part})"

    @property
    def ufl_shape(self):
        """Return the associated UFL shape."""
        return self._ufl_shape

    def ufl_function_space(self):
        """Get the function space of this Argument."""
        return self._ufl_function_space

    def ufl_domain(self):
        """Return the UFL domain."""
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        """Return The UFL element."""
        return self._ufl_function_space.ufl_element()

    def number(self):
        """Return the Argument number."""
        return self._number

    def part(self):
        """Return the part."""
        return self._part

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # TODO: Should in principle do like with Coefficient,
        # but that may currently simplify away some arguments
        # we want to keep, or? See issue#13.
        # When we can annotate zero with arguments, we can change this.
        return False

    def ufl_domains(self):
        """Return UFL domains."""
        return self._ufl_function_space.ufl_domains()

    def _ufl_signature_data_(self, renumbering):
        """Signature data.

        Signature data for form arguments depend on the global numbering
        of the form arguments and domains.
        """
        fsdata = self._ufl_function_space._ufl_signature_data_(renumbering)
        return ("Argument", self._number, self._part, fsdata)

    def __str__(self):
        """Format as a string."""
        number = str(self._number)
        if len(number) == 1:
            s = f"v_{number}"
        else:
            s = f"v_{{{number}}}"
        if self._part is not None:
            part = str(self._part)
            if len(part) == 1:
                s = f"{s}^{part}"
            else:
                s = f"{s}^{{{part}}}"
        return s

    def __repr__(self):
        """Representation."""
        return self._repr

    def __eq__(self, other):
        """Check equality.

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
        return (
            type(self) is type(other)
            and self._number == other._number
            and self._part == other._part
            and self._ufl_function_space == other._ufl_function_space
        )


@ufl_type()
class Argument(FormArgument, BaseArgument):
    """UFL value: Representation of an argument to a form."""

    __slots__ = (
        "_number",
        "_part",
        "_repr",
        "_ufl_function_space",
        "_ufl_shape",
    )

    _primal = True
    _dual = False

    __getnewargs__ = BaseArgument.__getnewargs__
    __str__ = BaseArgument.__str__
    _ufl_signature_data_ = BaseArgument._ufl_signature_data_
    __eq__ = BaseArgument.__eq__

    def __new__(cls, *args, **kw):
        """Create new Argument."""
        if args[0] and is_dual(args[0]):
            return Coargument(*args, **kw)
        return super().__new__(cls)

    def __init__(self, function_space, number, part=None):
        """Initialise."""
        FormArgument.__init__(self)
        BaseArgument.__init__(self, function_space, number, part)

        self._repr = f"Argument({self._ufl_function_space!r}, {self._number!r}, {self._part!r})"

    def ufl_domains(self):
        """Return UFL domains."""
        return BaseArgument.ufl_domains(self)

    def __repr__(self):
        """Representation."""
        return self._repr


@ufl_type()
class Coargument(BaseForm, BaseArgument):
    """UFL value: Representation of an argument to a form in a dual space."""

    __slots__ = (
        "_arguments",
        "_coefficients",
        "_hash",
        "_number",
        "_part",
        "_repr",
        "_ufl_function_space",
        "_ufl_shape",
        "ufl_operands",
    )

    _primal = False
    _dual = True

    def __new__(cls, *args, **kw):
        """Create a new Coargument."""
        if args[0] and is_primal(args[0]):
            raise ValueError(
                "ufl.Coargument takes in a dual space! If you want to define an argument "
                "in the primal space you should use ufl.Argument."
            )
        return super().__new__(cls)

    def __init__(self, function_space, number, part=None):
        """Initialise."""
        BaseArgument.__init__(self, function_space, number, part)
        BaseForm.__init__(self)

        self.ufl_operands = ()
        self._hash = None
        self._repr = f"Coargument({self._ufl_function_space!r}, {self._number!r}, {self._part!r})"

    def arguments(self, outer_form=None):
        """Return all Argument objects found in form."""
        if self._arguments is None:
            self._analyze_form_arguments(outer_form=outer_form)
        return self._arguments

    def _analyze_form_arguments(self, outer_form=None):
        """Analyze which Argument and Coefficient objects can be found in the form."""
        # Define canonical numbering of arguments and coefficients
        self._coefficients = ()
        # Coarguments map from V* to V*, i.e. V* -> V*, or equivalently V* x V -> R.
        # So they have one argument in the primal space and one in the dual space.
        # However, when they are composed with linear forms with dual
        # arguments, such as BaseFormOperators, the primal argument is
        # discarded when analysing the argument as Coarguments.
        if not outer_form:
            self._arguments = (Argument(self.ufl_function_space().dual(), 0), self)
        else:
            self._arguments = (self,)

    def ufl_domain(self):
        """Return the UFL domain."""
        return BaseArgument.ufl_domain(self)

    def equals(self, other):
        """Check equality."""
        if type(other) is not Coargument:
            return False
        if self is other:
            return True
        return (
            self._ufl_function_space == other._ufl_function_space
            and self._number == other._number
            and self._part == other._part
        )

    def __hash__(self):
        """Hash."""
        return hash(("Coargument", hash(self._ufl_function_space), self._number, self._part))


# --- Helper functions for pretty syntax ---


def TestFunction(function_space, part=None):
    """UFL value: Create a test function argument to a form."""
    return Argument(function_space, 0, part)


def TrialFunction(function_space, part=None):
    """UFL value: Create a trial function argument to a form."""
    return Argument(function_space, 1, part)


# --- Helper functions for creating subfunctions on mixed elements ---


def Arguments(function_space, number):
    """Create an Argument in a mixed space.

    Returns a tuple with the function components corresponding to the subelements.
    """
    if isinstance(function_space, MixedFunctionSpace):
        return [
            Argument(function_space.ufl_sub_space(i), number, i)
            for i in range(function_space.num_sub_spaces())
        ]
    else:
        return split(Argument(function_space, number))


def TestFunctions(function_space):
    """Create a TestFunction in a mixed space.

    Returns a tuple with the function components corresponding to the
    subelements.
    """
    return Arguments(function_space, 0)


def TrialFunctions(function_space):
    """Create a TrialFunction in a mixed space.

    Returns a tuple with the function components corresponding to the
    subelements.
    """
    return Arguments(function_space, 1)
