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
# Modified by Massimiliano Leoni, 2016

# import six
from ufl.utils.py23 import as_native_str
from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.log import error

valid_restriction_domains = ("interior", "facet", "face", "edge", "vertex")


# @six.python_2_unicode_compatible
class RestrictedElement(FiniteElementBase):
    "Represents the restriction of a finite element to a type of cell entity."
    def __init__(self, element, restriction_domain):
        if not isinstance(element, FiniteElementBase):
            error("Expecting a finite element instance.")
        if restriction_domain not in valid_restriction_domains:
            error("Expecting one of the strings %s." % repr(valid_restriction_domains))

        FiniteElementBase.__init__(self, "RestrictedElement", element.cell(),
                                   element.degree(),
                                   element.quadrature_scheme(),
                                   element.value_shape(),
                                   element.reference_value_shape())

        self._element = element

        self._restriction_domain = restriction_domain

        self._repr = as_native_str("RestrictedElement(%s, %s)" % (
            repr(self._element), repr(self._restriction_domain)))

    def is_cellwise_constant(self):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        return self._element.is_cellwise_constant()

    def sub_element(self):
        "Return the element which is restricted."
        return self._element

    #def element(self):
    #    "Deprecated."
    #    deprecate("RestrictedElement.element() is deprecated, please use .sub_element() instead.")
    #    return self.sub_element()

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
        return "<%s>|_{%s}" % (self._element.shortstr(),
                               self._restriction_domain)

    def symmetry(self):
        """Return the symmetry dict, which is a mapping :math:`c_0 \\to c_1`
        meaning that component :math:`c_0` is represented by component
        :math:`c_1`.
        A component is a tuple of one or more ints."""
        return self._element.symmetry()

    def num_sub_elements(self):
        "Return number of sub elements."
        return self._element.num_sub_elements()

    def sub_elements(self):
        "Return list of sub elements."
        return self._element.sub_elements()

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
