"""This module defines the Coefficient class and a number of related classes, including Constant."""

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
import warnings

from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import FormArgument
from ufl.argument import Argument
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace, MixedFunctionSpace
from ufl.form import BaseForm
from ufl.split_functions import split
from ufl.utils.counted import Counted
from ufl.duals import is_primal, is_dual


# --- The Coefficient class represents a coefficient in a form ---


class BaseCoefficient(Counted):
    """UFL form argument type: Parent Representation of a form coefficient."""

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    # __slots__ = ("_count", "_ufl_function_space", "_repr", "_ufl_shape")
    _ufl_noslots_ = True
    __slots__ = ()
    _ufl_is_abstract_ = True

    def __getnewargs__(self):
        """Get new args."""
        return (self._ufl_function_space, self._count)

    def __init__(self, function_space, count=None):
        """Initalise."""
        Counted.__init__(self, count, Coefficient)

        if isinstance(function_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map
            # the cell to The Default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
            warnings.warn("The use of FiniteElement as an input to Coefficient will be deprecated by December 2023. "
                          "Please, use FunctionSpace instead", FutureWarning)
        elif not isinstance(function_space, AbstractFunctionSpace):
            raise ValueError("Expecting a FunctionSpace or FiniteElement.")

        self._ufl_function_space = function_space
        self._ufl_shape = function_space.ufl_element().value_shape()

        self._repr = "BaseCoefficient(%s, %s)" % (
            repr(self._ufl_function_space), repr(self._count))

    @property
    def ufl_shape(self):
        """Return the associated UFL shape."""
        return self._ufl_shape

    def ufl_function_space(self):
        """Get the function space of this coefficient."""
        return self._ufl_function_space

    def ufl_domain(self):
        """Shortcut to get the domain of the function space of this coefficient."""
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        """Shortcut to get the finite element of the function space of this coefficient."""
        return self._ufl_function_space.ufl_element()

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        return self.ufl_element().is_cellwise_constant()

    def ufl_domains(self):
        """Return tuple of domains related to this terminal object."""
        return self._ufl_function_space.ufl_domains()

    def _ufl_signature_data_(self, renumbering):
        """Signature data for form arguments depend on the global numbering of the form arguments and domains."""
        count = renumbering[self]
        fsdata = self._ufl_function_space._ufl_signature_data_(renumbering)
        return ("Coefficient", count, fsdata)

    def __str__(self):
        """Format as a string."""
        return f"w_{self._count}"

    def __repr__(self):
        """Representation."""
        return self._repr

    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, BaseCoefficient):
            return False
        if self is other:
            return True
        return self._count == other._count and self._ufl_function_space == other._ufl_function_space


@ufl_type()
class Cofunction(BaseCoefficient, BaseForm):
    """UFL form argument type: Representation of a form coefficient from a dual space."""

    __slots__ = (
        "_count",
        "_counted_class",
        "_arguments",
        "_coefficients",
        "_ufl_function_space",
        "ufl_operands",
        "_repr",
        "_ufl_shape",
        "_hash"
    )
    _primal = False
    _dual = True

    def __new__(cls, *args, **kw):
        """Create a new Cofunction."""
        if args[0] and is_primal(args[0]):
            raise ValueError("ufl.Cofunction takes in a dual space. If you want to define a coefficient "
                             "in the primal space you should use ufl.Coefficient.")
        return super().__new__(cls)

    def __init__(self, function_space, count=None):
        """Initialise."""
        BaseCoefficient.__init__(self, function_space, count)
        BaseForm.__init__(self)

        self.ufl_operands = ()
        self._hash = None
        self._repr = "Cofunction(%s, %s)" % (
            repr(self._ufl_function_space), repr(self._count))

    def equals(self, other):
        """Check equality."""
        if type(other) is not Cofunction:
            return False
        if self is other:
            return True
        return self._count == other._count and self._ufl_function_space == other._ufl_function_space

    def __hash__(self):
        """Hash."""
        return hash(("Cofunction",
                     hash(self._ufl_function_space),
                     self._count))

    def _analyze_form_arguments(self):
        """Analyze which Argument and Coefficient objects can be found in the form."""
        # Define canonical numbering of arguments and coefficients
        # Cofunctions have one argument in primal space as they map from V to R.
        self._arguments = (Argument(self._ufl_function_space.dual(), 0),)
        self._coefficients = (self,)


@ufl_type()
class Coefficient(FormArgument, BaseCoefficient):
    """UFL form argument type: Representation of a form coefficient."""

    _ufl_noslots_ = True
    _primal = True
    _dual = False

    __getnewargs__ = BaseCoefficient.__getnewargs__
    __str__ = BaseCoefficient.__str__
    _ufl_signature_data_ = BaseCoefficient._ufl_signature_data_

    def __new__(cls, *args, **kw):
        """Create a new Coefficient."""
        if args[0] and is_dual(args[0]):
            return Cofunction(*args, **kw)
        return super().__new__(cls)

    def __init__(self, function_space, count=None):
        """Initialise."""
        FormArgument.__init__(self)
        BaseCoefficient.__init__(self, function_space, count)

        self._repr = "Coefficient(%s, %s)" % (
            repr(self._ufl_function_space), repr(self._count))

    def ufl_domains(self):
        """Get the UFL domains."""
        return BaseCoefficient.ufl_domains(self)

    def __eq__(self, other):
        """Check equality."""
        if not isinstance(other, Coefficient):
            return False
        if self is other:
            return True
        return self._count == other._count and self._ufl_function_space == other._ufl_function_space

    def __repr__(self):
        """Representation."""
        return self._repr


# --- Helper functions for subfunctions on mixed elements ---

def Coefficients(function_space):
    """Create a Coefficient in a mixed space.

    Returns a tuple with the function components corresponding to the subelements.
    """
    if isinstance(function_space, MixedFunctionSpace):
        return [Coefficient(fs) if is_primal(fs) else Cofunction(fs)
                for fs in function_space.num_sub_spaces()]
    else:
        return split(Coefficient(function_space))
