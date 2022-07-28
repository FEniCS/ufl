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

from itertools import chain

from ufl.log import error
from ufl.cell import TensorProductCell, as_cell
from ufl.sobolevspace import DirectionalSobolevSpace, H1, H2

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class TensorProductElement(FiniteElementBase):
    r"""The tensor product of :math:`d` element spaces:

    .. math:: V = V_1 \otimes V_2 \otimes ...  \otimes V_d

    Given bases :math:`\{\phi_{j_i}\}` of the spaces :math:`V_i` for :math:`i = 1, ...., d`,
    :math:`\{ \phi_{j_1} \otimes \phi_{j_2} \otimes \cdots \otimes \phi_{j_d}
    \}` forms a basis for :math:`V`.
    """
    __slots__ = ("_sub_elements", "_cell")

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
        elif all(e.mapping() == "L2 Piola" for e in self._sub_elements):
            return "L2 Piola"
        else:
            return "undefined"

    def sobolev_space(self):
        "Return the underlying Sobolev space of the TensorProductElement."
        elements = self._sub_elements
        if all(e.sobolev_space() == elements[0].sobolev_space()
               for e in elements):
            return elements[0].sobolev_space()
        else:
            # Generate a DirectionalSobolevSpace which contains
            # continuity information parametrized by spatial index
            orders = []
            for e in elements:
                e_dim = e.cell().geometric_dimension()
                e_order = (e.sobolev_space()._order,) * e_dim
                orders.extend(e_order)
            return DirectionalSobolevSpace(orders)

    def num_sub_elements(self):
        "Return number of subelements."
        return len(self._sub_elements)

    def sub_elements(self):
        "Return subelements (factors)."
        return self._sub_elements

    def reconstruct(self, **kwargs):
        cell = kwargs.pop("cell", self.cell())
        return TensorProductElement(*[e.reconstruct(**kwargs) for e in self.sub_elements()], cell=cell)

    def __str__(self):
        "Pretty-print."
        return "TensorProductElement(%s, cell=%s)" \
            % (', '.join([str(e) for e in self._sub_elements]), str(self._cell))

    def shortstr(self):
        "Short pretty-print."
        return "TensorProductElement(%s, cell=%s)" \
            % (', '.join([e.shortstr() for e in self._sub_elements]), str(self._cell))

    def is_fully_continuous(self):
        """Return true if the values of this element's basis functions are continuous between elements."""
        return self.sobolev_space() in [H1, H2]
