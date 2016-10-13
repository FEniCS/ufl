# -*- coding: utf-8 -*-
# Copyright (C) 2008-2016 Andrew T. T. McRae
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
# Modified by Massimiliano Leoni, 2016

# import six
from ufl.utils.py23 import as_native_str
from ufl.utils.py23 import as_native_strings
from ufl.finiteelement.finiteelementbase import FiniteElementBase


# @six.python_2_unicode_compatible
class HDivElement(FiniteElementBase):
    """A div-conforming version of an outer product element, assuming
    this makes mathematical sense."""
    __slots__ = as_native_strings(("_element",))

    def __init__(self, element):
        self._element = element
        self._repr = as_native_str("HDivElement(%s)" % repr(element))

        family = "TensorProductElement"
        cell = element.cell()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = (element.cell().geometric_dimension(),)
        reference_value_shape = (element.cell().topological_dimension(),)

        # Skipping TensorProductElement constructor! Bad code smell, refactor to avoid this non-inheritance somehow.
        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def mapping(self):
        return "contravariant Piola"

    def __str__(self):
        return "HDivElement(%s)" % str(self._element)

    def shortstr(self):
        "Format as string for pretty printing."
        return "HDivElement(%s)" % str(self._element.shortstr())


# @six.python_2_unicode_compatible
class HCurlElement(FiniteElementBase):
    """A curl-conforming version of an outer product element, assuming
    this makes mathematical sense."""
    __slots__ = as_native_strings(("_element",))

    def __init__(self, element):
        self._element = element
        self._repr = as_native_str("HCurlElement(%s)" % repr(element))

        family = "TensorProductElement"
        cell = element.cell()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        cell = element.cell()
        value_shape = (cell.geometric_dimension(),)
        reference_value_shape = (cell.topological_dimension(),)  # TODO: Is this right?
        # Skipping TensorProductElement constructor! Bad code smell,
        # refactor to avoid this non-inheritance somehow.
        FiniteElementBase.__init__(self, family, cell, degree, quad_scheme,
                                   value_shape, reference_value_shape)

    def mapping(self):
        return "covariant Piola"

    def __str__(self):
        return "HCurlElement(%s)" % str(self._element)

    def shortstr(self):
        "Format as string for pretty printing."
        return "HCurlElement(%s)" % str(self._element.shortstr())
