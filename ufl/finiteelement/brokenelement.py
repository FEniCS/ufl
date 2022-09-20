# -*- coding: utf-8 -*-
# Copyright (C) 2014 Andrew T. T. McRae
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

from ufl.finiteelement.finiteelementbase import FiniteElementBase


class BrokenElement(FiniteElementBase):
    """The discontinuous version of an existing Finite Element space."""
    def __init__(self, element):
        self._element = element

        family = "BrokenElement"
        cell = element.cell()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = element.value_shape()
        reference_value_shape = element.reference_value_shape()
        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def __repr__(self):
        return f"BrokenElement({repr(self._element)})"

    def mapping(self):
        return self._element.mapping()

    def reconstruct(self, **kwargs):
        return BrokenElement(self._element.reconstruct(**kwargs))

    def __str__(self):
        return f"BrokenElement({repr(self._element)})"

    def shortstr(self):
        """Format as string for pretty printing."""
        return f"BrokenElement({repr(self._element)})"
