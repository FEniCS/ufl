# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

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
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012

from six.moves import zip
from ufl.assertions import ufl_assert
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.finiteelementbase import FiniteElementBase

class EnrichedElement(FiniteElementBase):
    """The vector sum of two finite element spaces:

        EnrichedElement(V, Q) = {v + q | v in V, q in Q}.
    """
    def __init__(self, *elements):
        self._elements = elements

        cell = elements[0].cell()
        ufl_assert(all(e.cell() == cell for e in elements),
                   "Cell mismatch for sub elements of enriched element.")

        if isinstance(elements[0].degree(), int):
            degrees = { e.degree() for e in elements } - { None }
            degree = max(degrees) if degrees else None
        else:
            degree = tuple(map(max, zip(*[e.degree() for e in elements])))

        # We can allow the scheme not to be defined, but all defined should be equal
        quad_schemes = [e.quadrature_scheme() for e in elements]
        quad_schemes = [qs for qs in quad_schemes if qs is not None]
        quad_scheme = quad_schemes[0] if quad_schemes else None
        ufl_assert(all(qs == quad_scheme for qs in quad_schemes),\
            "Quadrature scheme mismatch.")

        value_shape = elements[0].value_shape()
        ufl_assert(all(e.value_shape() == value_shape for e in elements),
                   "Element value shape mismatch.")

        reference_value_shape = elements[0].reference_value_shape()
        ufl_assert(all(e.reference_value_shape() == reference_value_shape for e in elements),
                   "Element reference value shape mismatch.")

        #mapping = elements[0].mapping() # FIXME: This fails for a mixed subelement here.
        #ufl_assert(all(e.mapping() == mapping for e in elements),
        #           "Element mapping mismatch.")

        # Initialize element data
        FiniteElementBase.__init__(self, "EnrichedElement", cell, degree,
                                   quad_scheme, value_shape, reference_value_shape)

        # Cache repr string
        self._repr = "EnrichedElement(*%r)" % ([repr(e) for e in self._elements],)

    def reconstruct(self, **kwargs):
        """Construct a new EnrichedElement object with some properties
        replaced with new values."""
        elements = [e.reconstruct(**kwargs) for e in self._elements]
        if all(a == b for (a, b) in zip(elements, self._elements)):
            return self
        return EnrichedElement(*elements)

    def is_cellwise_constant(self):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return all(e.is_cellwise_constant() for e in self._elements)

    def mapping(self):
        return self._elements[0].mapping()

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(str(e) for e in self._elements)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(e.shortstr() for e in self._elements)
