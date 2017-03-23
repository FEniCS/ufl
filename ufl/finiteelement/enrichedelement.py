# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

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
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Massimiliano Leoni, 2016

# import six
from ufl.utils.py23 import as_native_str
from six.moves import zip
from ufl.log import error
from ufl.finiteelement.finiteelementbase import FiniteElementBase


# @six.python_2_unicode_compatible
class EnrichedElementBase(FiniteElementBase):
    """The vector sum of several finite element spaces:

        .. math:: \\textrm{EnrichedElement}(V, Q) = \\{v + q | v \\in V, q \\in Q\\}.
    """
    def __init__(self, *elements):
        self._elements = elements

        cell = elements[0].cell()
        if not all(e.cell() == cell for e in elements[1:]):
            error("Cell mismatch for sub elements of enriched element.")

        if isinstance(elements[0].degree(), int):
            degrees = {e.degree() for e in elements} - {None}
            degree = max(degrees) if degrees else None
        else:
            degree = tuple(map(max, zip(*[e.degree() for e in elements])))

        # We can allow the scheme not to be defined, but all defined
        # should be equal
        quad_schemes = [e.quadrature_scheme() for e in elements]
        quad_schemes = [qs for qs in quad_schemes if qs is not None]
        quad_scheme = quad_schemes[0] if quad_schemes else None
        if not all(qs == quad_scheme for qs in quad_schemes):
            error("Quadrature scheme mismatch.")

        value_shape = elements[0].value_shape()
        if not all(e.value_shape() == value_shape for e in elements[1:]):
            error("Element value shape mismatch.")

        reference_value_shape = elements[0].reference_value_shape()
        if not all(e.reference_value_shape() == reference_value_shape for e in elements[1:]):
            error("Element reference value shape mismatch.")

        # mapping = elements[0].mapping() # FIXME: This fails for a mixed subelement here.
        # if not all(e.mapping() == mapping for e in elements[1:]):
        #    error("Element mapping mismatch.")

        # Get name of subclass: EnrichedElement or NodalEnrichedElement
        class_name = as_native_str(self.__class__.__name__)

        # Initialize element data
        FiniteElementBase.__init__(self, class_name, cell, degree,
                                   quad_scheme, value_shape,
                                   reference_value_shape)

        # Cache repr string
        self._repr = as_native_str("%s(%s)" %
            (class_name, ", ".join(repr(e) for e in self._elements)))

    def mapping(self):
        return self._elements[0].mapping()

    def sobolev_space(self):
        "Return the underlying Sobolev space of the EnrichedElement"
        elements = [e for e in self._elements]
        if all(e.sobolev_space() == elements[0].sobolev_space()
               for e in elements):
            return elements[0].sobolev_space()
        else:
            # Find smallest shared Sobolev space over all sub elements
            spaces = [e.sobolev_space() for e in elements]
            superspaces = [{s} | set(s.parents) for s in spaces]
            intersect = set.intersection(*superspaces)
            for s in intersect.copy():
                for parent in s.parents:
                    intersect.discard(parent)

            sobolev_space, = intersect
            return sobolev_space

    def reconstruct(self, **kwargs):
        return type(self)(*[e.reconstruct(**kwargs) for e in self._elements])


class EnrichedElement(EnrichedElementBase):
    """The vector sum of several finite element spaces:

        .. math:: \\textrm{EnrichedElement}(V, Q) = \\{v + q | v \\in V, q \\in Q\\}.

        Dual basis is a concatenation of subelements dual bases;
        primal basis is a concatenation of subelements primal bases;
        resulting element is not nodal even when subelements are.
        Structured basis may be exploited in form compilers.
    """
    def is_cellwise_constant(self):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return all(e.is_cellwise_constant() for e in self._elements)

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(str(e) for e in self._elements)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(e.shortstr() for e in self._elements)


class NodalEnrichedElement(EnrichedElementBase):
    """The vector sum of several finite element spaces:

        .. math:: \\textrm{EnrichedElement}(V, Q) = \\{v + q | v \\in V, q \\in Q\\}.

        Primal basis is reorthogonalized to dual basis which is
        a concatenation of subelements dual bases; resulting
        element is nodal.
    """
    def is_cellwise_constant(self):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return False

    def __str__(self):
        "Format as string for pretty printing."
        return "<Nodal enriched element(%s)>" % ", ".join(str(e) for e in self._elements)

    def shortstr(self):
        "Format as string for pretty printing."
        return "NodalEnriched(%s)" % ", ".join(e.shortstr() for e in self._elements)
