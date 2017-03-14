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
from itertools import chain

from ufl.log import error
from ufl.utils.py23 import as_native_strings
from ufl.cell import TensorProductCell, as_cell

from ufl.finiteelement.finiteelementbase import FiniteElementBase


# @six.python_2_unicode_compatible
class TensorProductElement(FiniteElementBase):
    r"""The tensor product of :math:`d` element spaces:

    .. math:: V = V_1 \otimes V_2 \otimes ...  \otimes V_d

    Given bases :math:`\{\phi_{j_i}\}` of the spaces :math:`V_i` for :math:`i = 1, ...., d`,
    :math:`\{ \phi_{j_1} \otimes \phi_{j_2} \otimes \cdots \otimes \phi_{j_d}
    \}` forms a basis for :math:`V`.
    """
    __slots__ = as_native_strings(("_sub_elements", "_cell"))

    def __init__(self, *elements, **kwargs):
        "Create TensorProductElement from a given list of elements."
        if not elements:
            error("Cannot create TensorProductElement from empty list.")

        keywords = list(kwargs.keys())
        if keywords and keywords != ["cell"]:
            raise ValueError("TensorProductElement got an unexpected keyword argument '%s'" % keywords[0])
        cell = kwargs.get("cell")

        family = "TensorProductElement"

        if cell is None:
            # Define cell as the product of each elements cell
            cell = TensorProductCell(*[e.cell() for e in elements])
        else:
            cell = as_cell(cell)

        # Define polynomial degree as a tuple of sub-degrees
        degree = tuple(e.degree() for e in elements)

        # No quadrature scheme defined
        quad_scheme = None

        # match FIAT implementation
        value_shape = tuple(chain(*[e.value_shape() for e in elements]))
        reference_value_shape = tuple(chain(*[e.reference_value_shape() for e in elements]))
        if len(value_shape) > 1:
            error("Product of vector-valued elements not supported")
        if len(reference_value_shape) > 1:
            error("Product of vector-valued elements not supported")

        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, value_shape,
                                   reference_value_shape)
        self._sub_elements = elements
        self._cell = cell
        self._repr = "TensorProductElement(%s, cell=%s)" % (
            ", ".join(repr(e) for e in elements), repr(cell))

    def mapping(self):
        if all(e.mapping() == "identity" for e in self._sub_elements):
            return "identity"
        else:
            return "undefined"

    def sobolev_space(self):
        "Return the underlying Sobolev space of the TensorProductElement."
        elements = self._sub_elements
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
                    if parent in intersect:
                        intersect.remove(parent)

            sobolev_space, = intersect
            return sobolev_space

    def num_sub_elements(self):
        "Return number of subelements."
        return len(self._sub_elements)

    def sub_elements(self):
        "Return subelements (factors)."
        return self._sub_elements

    def reconstruct(self, cell=None):
        return TensorProductElement(*self.sub_elements(), cell=cell)

    def __str__(self):
        "Pretty-print."
        return "TensorProductElement(%s, cell=%s)" \
            % (', '.join([str(e) for e in self._sub_elements]), str(self._cell))

    def shortstr(self):
        "Short pretty-print."
        return "TensorProductElement(%s, cell=%s)" \
            % (', '.join([e.shortstr() for e in self._sub_elements]), str(self._cell))
