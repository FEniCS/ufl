# -*- coding: utf-8 -*-
# Copyright (C) 2008-2016 Andrew T. T. McRae
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016

from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.sobolevspace import HDiv, HCurl


class HDivElement(FiniteElementBase):
    """A div-conforming version of an outer product element, assuming
    this makes mathematical sense."""
    __slots__ = ("_element",)

    def __init__(self, element):
        self._element = element
        self._repr = "HDivElement(%s)" % repr(element)

        family = "TensorProductElement"
        cell = element.cell()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        value_shape = (element.cell().geometric_dimension(),)
        reference_value_shape = (element.cell().topological_dimension(),)

        # Skipping TensorProductElement constructor! Bad code smell, refactor to avoid this non-inheritance somehow.
        FiniteElementBase.__init__(self, family, cell, degree,
                                   quad_scheme, value_shape, reference_value_shape)

    def mapping(self):
        return "contravariant Piola"

    def sobolev_space(self):
        "Return the underlying Sobolev space."
        return HDiv

    def reconstruct(self, **kwargs):
        return HDivElement(self._element.reconstruct(**kwargs))

    def __str__(self):
        return "HDivElement(%s)" % str(self._element)

    def shortstr(self):
        "Format as string for pretty printing."
        return "HDivElement(%s)" % str(self._element.shortstr())


class HCurlElement(FiniteElementBase):
    """A curl-conforming version of an outer product element, assuming
    this makes mathematical sense."""
    __slots__ = ("_element",)

    def __init__(self, element):
        self._element = element
        self._repr = "HCurlElement(%s)" % repr(element)

        family = "TensorProductElement"
        cell = element.cell()
        degree = element.degree()
        quad_scheme = element.quadrature_scheme()
        cell = element.cell()
        value_shape = (cell.geometric_dimension(),)
        reference_value_shape = (cell.topological_dimension(),)  # TODO: Is this right?
        # Skipping TensorProductElement constructor! Bad code smell,
        # refactor to avoid this non-inheritance somehow.
        FiniteElementBase.__init__(self, family, cell, degree, quad_scheme,
                                   value_shape, reference_value_shape)

    def mapping(self):
        return "covariant Piola"

    def sobolev_space(self):
        "Return the underlying Sobolev space."
        return HCurl

    def reconstruct(self, **kwargs):
        return HCurlElement(self._element.reconstruct(**kwargs))

    def __str__(self):
        return "HCurlElement(%s)" % str(self._element)

    def shortstr(self):
        "Format as string for pretty printing."
        return "HCurlElement(%s)" % str(self._element.shortstr())


class WithMapping(FiniteElementBase):
    """Specify an alternative mapping for the wrappee. For example,
    to use identity mapping instead of Piola map with an element E,
    write
    remapped = WithMapping(E, "identity")
    """
    def __init__(self, wrapee, mapping):
        self._repr = "WithMapping(%s, %s)" % (repr(wrapee), mapping)
        self._mapping = mapping
        self.wrapee = wrapee

    def __getattr__(self, attr):
        try:
            return getattr(self.wrapee, attr)
        except AttributeError:
            raise AttributeError("'%s' object has no attribute '%s'" %
                                 (type(self).__name__, attr))

    def mapping(self):
        return self._mapping

    def reconstruct(self, **kwargs):
        mapping = kwargs.pop("mapping", self._mapping)
        wrapee = self.wrapee.reconstruct(**kwargs)
        return type(self)(wrapee, mapping)

    def __str__(self):
        return "WithMapping(%s, mapping=%s)" % (self.wrapee, self._mapping)

    def shortstr(self):
        return "WithMapping(%s, %s)" % (self.wrapee.shortstr(), self._mapping)
