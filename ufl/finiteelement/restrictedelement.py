# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Massimiliano Leoni, 2016

from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.sobolevspace import L2

valid_restriction_domains = ("interior", "facet", "face", "edge", "vertex")


class RestrictedElement(FiniteElementBase):
    "Represents the restriction of a finite element to a type of cell entity."

    def __init__(self, element, restriction_domain):
        if not isinstance(element, FiniteElementBase):
            raise ValueError("Expecting a finite element instance.")
        if restriction_domain not in valid_restriction_domains:
            raise ValueError(f"Expecting one of the strings: {valid_restriction_domains}")

        FiniteElementBase.__init__(self, "RestrictedElement", element.cell(),
                                   element.degree(),
                                   element.quadrature_scheme(),
                                   element.value_shape(),
                                   element.reference_value_shape())

        self._element = element

        self._restriction_domain = restriction_domain

    def __repr__(self):
        return f"RestrictedElement({repr(self._element)}, {repr(self._restriction_domain)})"

    def sobolev_space(self):
        return L2

    def is_cellwise_constant(self):
        """Return whether the basis functions of this element is spatially
        constant over each cell.

        """
        return self._element.is_cellwise_constant()

    def _is_linear(self):
        return self._element._is_linear()

    def sub_element(self):
        "Return the element which is restricted."
        return self._element

    def mapping(self):
        return self._element.mapping()

    def restriction_domain(self):
        "Return the domain onto which the element is restricted."
        return self._restriction_domain

    def reconstruct(self, **kwargs):
        element = self._element.reconstruct(**kwargs)
        return RestrictedElement(element, self._restriction_domain)

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
        :math:`c_1`.  A component is a tuple of one or more ints.

        """
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

    def variant(self):
        return self._element.variant()
