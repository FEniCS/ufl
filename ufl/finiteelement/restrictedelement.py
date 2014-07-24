"This module defines the UFL finite element classes."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, istr, EmptyDict
from ufl.geometry import Cell, as_cell
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.finiteelementbase import FiniteElementBase

class RestrictedElement(FiniteElementBase):
    "Represents the restriction of a finite element to a type of cell entity."
    def __init__(self, element, cell_restriction):
        ufl_assert(isinstance(element, FiniteElementBase),
                   "Expecting a finite element instance.")
        ufl_assert(isinstance(cell_restriction, Cell) or cell_restriction == "facet",
                   "Expecting a Cell instance, or the string 'facet'.")

        super(RestrictedElement, self).__init__("RestrictedElement", element.cell(),
            element.degree(), element.quadrature_scheme(), element.value_shape())
        self._element = element

        if isinstance(cell_restriction, Cell):
            # Just attach cell_restriction if it is a Cell
            self._cell_restriction = cell_restriction
        elif cell_restriction == "facet":
            # Create a cell
            cell = element.cell()
            self._cell_restriction = Cell(cell.facet_cellname(),
                                          geometric_dimension=cell.geometric_dimension())

        self._repr = "RestrictedElement(%r, %r)" % (self._element, self._cell_restriction)

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        return "RestrictedElement(%s, %r)" % (self._element.reconstruction_signature(), self._cell_restriction)

    def reconstruct(self, **kwargs):
        """Construct a new RestrictedElement object with
        some properties replaced with new values."""
        element = self._element.reconstruct(**kwargs)
        cell_restriction = kwargs.get("cell_restriction", self.cell_restriction())
        return RestrictedElement(element=element, cell_restriction=cell_restriction)

    def is_cellwise_constant(self):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return self._element.is_cellwise_constant()

    def element(self):
        "Return the element which is restricted."
        return self._element

    def cell_restriction(self):
        "Return the domain onto which the element is restricted."
        return self._cell_restriction

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element, self._cell_restriction)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element.shortstr(), self._cell_restriction)

    def symmetry(self):
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return self._element.symmetry()

    def num_sub_elements(self):
        "Return number of sub elements"
        return self._element.num_sub_elements()
        #return 1

    def sub_elements(self):
        "Return list of sub elements"
        return self._element.sub_elements()
        #return [self._element]

    def num_restricted_sub_elements(self):
        # FIXME: Use this where intended, for disambiguation
        #        w.r.t. different sub_elements meanings.
        "Return number of restricted sub elements."
        return 1

    def restricted_sub_elements(self):
        # FIXME: Use this where intended, for disambiguation
        #        w.r.t. different sub_elements meanings.
        "Return list of restricted sub elements."
        return (self._element,)

    def signature_data(self, renumbering):
        data = ("RestrictedElement", self._element.signature_data(renumbering),
                repr(self._cell_restriction)) # Note: I'm pretty sure repr is safe here but that may change
        return data
