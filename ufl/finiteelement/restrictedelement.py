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

from ufl.assertions import ufl_assert
from ufl.geometry import Cell, as_cell
from ufl.log import info_blue, warning, warning_blue, error, deprecate

from ufl.finiteelement.finiteelementbase import FiniteElementBase

valid_restriction_domains = ("interior", "facet", "face", "edge", "vertex")

class RestrictedElement(FiniteElementBase):
    "Represents the restriction of a finite element to a type of cell entity."
    def __init__(self, element, restriction_domain):
        ufl_assert(isinstance(element, FiniteElementBase),
                   "Expecting a finite element instance.")
        ufl_assert(restriction_domain in valid_restriction_domains,
                   "Expecting one of the strings %r." % (valid_restriction_domains,))

        FiniteElementBase.__init__(self, "RestrictedElement", element.ufl_domain(),
            element.degree(), element.quadrature_scheme(), element.value_shape(), element.reference_value_shape())

        self._element = element

        self._restriction_domain = restriction_domain

        self._repr = "RestrictedElement(%r, %r)" % (self._element, self._restriction_domain)

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        return "RestrictedElement(%s, %r)" % (self._element.reconstruction_signature(), self._restriction_domain)

    def reconstruct(self, **kwargs):
        """Construct a new RestrictedElement object with
        some properties replaced with new values."""
        element = self._element.reconstruct(**kwargs)
        restriction_domain = kwargs.get("restriction_domain", self.restriction_domain())
        return RestrictedElement(element=element, restriction_domain=restriction_domain)

    def is_cellwise_constant(self):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return self._element.is_cellwise_constant()

    def sub_element(self):
        "Return the element which is restricted."
        return self._element

    def element(self):
        deprecate("RestrictedElement.element() is deprecated, please use .sub_element() instead.")
        return self.sub_element()

    def mapping(self):
        return self._element.mapping()

    def restriction_domain(self):
        "Return the domain onto which the element is restricted."
        return self._restriction_domain

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element, self._restriction_domain)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element.shortstr(), self._restriction_domain)

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

    def _ufl_signature_data_(self, renumbering):
        data = ("RestrictedElement", self._element._ufl_signature_data_(renumbering), self._restriction_domain)
        return data
