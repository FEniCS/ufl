# -*- coding: utf-8 -*-
# Copyright (C) 2014 Andrew T. T. McRae
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

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class TraceElement(FiniteElementBase):
    """A finite element space: the trace of a given hdiv element.
    This is effectively a scalar-valued restriction which is
    non-zero only on cell facets."""
    def __init__(self, element):
        self._element = element
        self._repr = "TraceElement(%s)" % repr(element)

        family = "TraceElement"
        cell = element.cell()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = ()
        reference_value_shape = ()
        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def mapping(self):
        return "identity"

    def __str__(self):
        return "TraceElement(%s)" % str(self._element)

    def shortstr(self):
        return "TraceElement(%s)" % str(self._element.shortstr())

    def __repr__(self):
        return self._repr
