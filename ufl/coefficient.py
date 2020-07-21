# -*- coding: utf-8 -*-
"""This module defines the Coefficient class and a number
of related classes, including Constant."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009.
# Modified by Massimiliano Leoni, 2016.
# Modified by Cecile Daversin-Catty, 2018.

from ufl.log import error
from ufl.core.ufl_type import ufl_type
from ufl.core.terminal import Terminal, FormArgument
from ufl.finiteelement import FiniteElementBase
from ufl.domain import default_domain
from ufl.functionspace import AbstractFunctionSpace, FunctionSpace, MixedFunctionSpace
from ufl.split_functions import split
from ufl.utils.counted import counted_init

# --- The Coefficient class represents a coefficient in a form ---


@ufl_type()
class Coefficient(FormArgument):
    """UFL form argument type: Representation of a form coefficient."""

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    # __slots__ = ("_count", "_ufl_function_space", "_repr", "_ufl_shape")
    _ufl_noslots_ = True
    _globalcount = 0

    def __init__(self, function_space, count=None):
        FormArgument.__init__(self)
        counted_init(self, count, Coefficient)

        if isinstance(function_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map
            # the cell to The Default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, AbstractFunctionSpace):
            error("Expecting a FunctionSpace or FiniteElement.")

        self._ufl_function_space = function_space
        self._ufl_shape = function_space.ufl_element().value_shape()

        self._repr = "Coefficient(%s, %s)" % (
            repr(self._ufl_function_space), repr(self._count))

    def count(self):
        return self._count

    @property
    def ufl_shape(self):
        "Return the associated UFL shape."
        return self._ufl_shape

    def ufl_function_space(self):
        "Get the function space of this coefficient."
        return self._ufl_function_space

    def ufl_domain(self):
        "Shortcut to get the domain of the function space of this coefficient."
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        "Shortcut to get the finite element of the function space of this coefficient."
        return self._ufl_function_space.ufl_element()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return self.ufl_element().is_cellwise_constant()

    def ufl_domains(self):
        "Return tuple of domains related to this terminal object."
        return self._ufl_function_space.ufl_domains()

    def _ufl_signature_data_(self, renumbering):
        "Signature data for form arguments depend on the global numbering of the form arguments and domains."
        count = renumbering[self]
        fsdata = self._ufl_function_space._ufl_signature_data_(renumbering)
        return ("Coefficient", count, fsdata)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "w_%s" % count
        else:
            return "w_{%s}" % count

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, Coefficient):
            return False
        if self is other:
            return True
        return (self._count == other._count and
                self._ufl_function_space == other._ufl_function_space)


# --- Helper functions for subfunctions on mixed elements ---

def Coefficients(function_space):
    """UFL value: Create a Coefficient in a mixed space, and return a
    tuple with the function components corresponding to the subelements."""
    if isinstance(function_space, MixedFunctionSpace):
        return [Coefficient(function_space.ufl_sub_space(i))
                for i in range(function_space.num_sub_spaces())]
    else:
        return split(Coefficient(function_space))


# --- Subspace ---

@ufl_type()
class Subspace(Terminal):
    """UFL terminal type: Representation of a subspace."""

    __slots__ = ("_count", "_ufl_function_space", "_repr", "_ufl_shape")
    _globalcount = 0

    def __init__(self, function_space, count=None):
        Terminal.__init__(self)
        counted_init(self, count, Subspace)

        if isinstance(function_space, FiniteElementBase):
            # For legacy support for .ufl files using cells, we map
            # the cell to The Default Mesh
            element = function_space
            domain = default_domain(element.cell())
            function_space = FunctionSpace(domain, element)
        elif not isinstance(function_space, FunctionSpace):
            error("Expecting a FunctionSpace.")

        self._ufl_function_space = function_space
        self._ufl_shape = function_space.ufl_element().value_shape()

        self._repr = "Subspace(%s, %s)" % (
            repr(self._ufl_function_space), repr(self._count))

    def count(self):
        return self._count

    @property
    def ufl_shape(self):
        "Return the associated UFL shape."
        return self._ufl_shape

    def ufl_function_space(self):
        "Get the function space of this subspace."
        return self._ufl_function_space

    def ufl_domain(self):
        "Shortcut to get the domain of the function space of this subspace."
        return self._ufl_function_space.ufl_domain()

    def ufl_element(self):
        "Shortcut to get the finite element of the function space of this subspace."
        return self._ufl_function_space.ufl_element()

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return self.ufl_element().is_cellwise_constant()

    def ufl_domains(self):
        "Return tuple of domains related to this terminal object."
        return self._ufl_function_space.ufl_domains()

    def _ufl_signature_data_(self, renumbering):
        "Signature data depend on the global numbering of the subspace and domains."
        count = renumbering[self]
        fsdata = self._ufl_function_space._ufl_signature_data_(renumbering)
        return ("Subspace", count, fsdata)

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "t_%s" % count
        else:
            return "t_{%s}" % count

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self is other:
            return True
        return (self._count == other._count and
                self._ufl_function_space == other._ufl_function_space)
