"""This module defines the UFL finite element classes."""

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


class EnrichedElementBase(FiniteElementBase):
    """The vector sum of several finite element spaces."""

    def __init__(self, *elements):
        """Doc."""
        self._elements = elements

        cell = elements[0].cell()
        if not all(e.cell() == cell for e in elements[1:]):
            raise ValueError("Cell mismatch for sub elements of enriched element.")

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
            raise ValueError("Quadrature scheme mismatch.")

        value_shape = elements[0].value_shape()
        if not all(e.value_shape() == value_shape for e in elements[1:]):
            raise ValueError("Element value shape mismatch.")

        reference_value_shape = elements[0].reference_value_shape()
        if not all(e.reference_value_shape() == reference_value_shape for e in elements[1:]):
            raise ValueError("Element reference value shape mismatch.")

        # mapping = elements[0].mapping() # FIXME: This fails for a mixed subelement here.
        # if not all(e.mapping() == mapping for e in elements[1:]):
        #    raise ValueError("Element mapping mismatch.")

        # Get name of subclass: EnrichedElement or NodalEnrichedElement
        class_name = self.__class__.__name__

        # Initialize element data
        FiniteElementBase.__init__(self, class_name, cell, degree,
                                   quad_scheme, value_shape,
                                   reference_value_shape)

    def mapping(self):
        """Doc."""
        return self._elements[0].mapping()

    def sobolev_space(self):
        """Return the underlying Sobolev space."""
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

    def variant(self):
        """Doc."""
        try:
            variant, = {e.variant() for e in self._elements}
            return variant
        except ValueError:
            return None

    def reconstruct(self, **kwargs):
        """Doc."""
        return type(self)(*[e.reconstruct(**kwargs) for e in self._elements])


class EnrichedElement(EnrichedElementBase):
    r"""The vector sum of several finite element spaces.

    .. math:: \\textrm{EnrichedElement}(V, Q) = \\{v + q | v \\in V, q \\in Q\\}.

    Dual basis is a concatenation of subelements dual bases;
    primal basis is a concatenation of subelements primal bases;
    resulting element is not nodal even when subelements are.
    Structured basis may be exploited in form compilers.
    """

    def is_cellwise_constant(self):
        """Return whether the basis functions of this element is spatially constant over each cell."""
        return all(e.is_cellwise_constant() for e in self._elements)

    def __repr__(self):
        """Doc."""
        return "EnrichedElement(" + ", ".join(repr(e) for e in self._elements) + ")"

    def __str__(self):
        """Format as string for pretty printing."""
        return "<%s>" % " + ".join(str(e) for e in self._elements)

    def shortstr(self):
        """Format as string for pretty printing."""
        return "<%s>" % " + ".join(e.shortstr() for e in self._elements)


class NodalEnrichedElement(EnrichedElementBase):
    r"""The vector sum of several finite element spaces.

    .. math:: \\textrm{EnrichedElement}(V, Q) = \\{v + q | v \\in V, q \\in Q\\}.

    Primal basis is reorthogonalized to dual basis which is
    a concatenation of subelements dual bases; resulting
    element is nodal.
    """
    def is_cellwise_constant(self):
        """Return whether the basis functions of this element is spatially constant over each cell."""
        return False

    def __repr__(self):
        """Doc."""
        return "NodalEnrichedElement(" + ", ".join(repr(e) for e in self._elements) + ")"

    def __str__(self):
        """Format as string for pretty printing."""
        return "<Nodal enriched element(%s)>" % ", ".join(str(e) for e in self._elements)

    def shortstr(self):
        """Format as string for pretty printing."""
        return "NodalEnriched(%s)" % ", ".join(e.shortstr() for e in self._elements)
