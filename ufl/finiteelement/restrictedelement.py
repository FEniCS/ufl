"This module defines the UFL finite element classes."

# Copyright (C) 2008-2012 Martin Sandve Alnes
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
#
# First added:  2008-03-03
# Last changed: 2012-08-16

from itertools import izip
from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, index_to_component, component_to_index, istr, EmptyDict
from ufl.geometry import as_cell, domain2facet
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.finiteelementbase import FiniteElementBase

class RestrictedElement(FiniteElementBase):
    def __init__(self, element, domain):
        ufl_assert(isinstance(element, FiniteElementBase), "Expecting a finite element instance.")
        from ufl.integral import Measure
        from ufl.geometry import Cell
        ufl_assert(isinstance(domain, Measure) or domain == "facet"\
                   or isinstance(as_cell(domain), Cell),\
            "Expecting a subdomain represented by a Measure, a Cell instance, or the string 'facet'.")
        super(RestrictedElement, self).__init__("RestrictedElement", element.cell(),\
            element.degree(), element.quadrature_scheme(), element.value_shape())
        self._element = element

        # Just attach domain if it is a Measure or Cell
        if isinstance(domain, (Measure, Cell)):
            self._domain = domain
        else:
            # Check for facet and handle it
            if domain == "facet":
                cell = self.cell()
                ufl_assert(not cell.is_undefined(), "Cannot determine facet cell of undefined cell.")
                domain = Cell(domain2facet[cell.domain()])
            else:
                # Create Cell (if we get a string)
                domain = as_cell(domain)
            self._domain = domain

        if isinstance(self._domain, Cell) and self._domain.is_undefined():
            warning("Undefined cell as domain in RestrictedElement. Not sure if this is well defined.")

        # Cache repr string
        self._repr = "RestrictedElement(%r, %r)" % (self._element, self._domain)

    def reconstruct(self, **kwargs):
        """Construct a new RestrictedElement object with some properties
        replaced with new values."""
        element = self._element.reconstruct(**kwargs)
        domain = kwargs.get("domain", self.domain())
        return RestrictedElement(element=element, domain=domain)

    def is_cellwise_constant(self):
        "Return whether the basis functions of this element is spatially constant over each cell."
        return self._element.is_cellwise_constant()

    def element(self):
        "Return the element which is restricted."
        return self._element

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element, self._domain)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element.shortstr(), self._domain)

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

    def num_restricted_sub_elements(self): # FIXME: Use this where intended, for disambiguation w.r.t. different sub_elements meanings.
        "Return number of restricted sub elements."
        return 1

    def restricted_sub_elements(self): # FIXME: Use this where intended, for disambiguation w.r.t. different sub_elements meanings.
        "Return list of restricted sub elements."
        return (self._element,)
