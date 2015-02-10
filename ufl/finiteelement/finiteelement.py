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
# Modified by Anders Logg 2014

from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, istr, EmptyDict
from ufl.geometry import as_domain, as_cell
from ufl.log import info_blue, warning, warning_blue, error

from ufl.cell import OuterProductCell
from ufl.finiteelement.elementlist import canonical_element_description
from ufl.finiteelement.finiteelementbase import FiniteElementBase

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"
    # TODO: Move these to base?
    __slots__ = ("_short_name",
                 "_sobolev_space",
                 "_mapping",
                )

    def __new__(cls,
                family,
                domain=None,
                degree=None,
                form_degree=None,
                quad_scheme=None):
        """Intercepts construction to expand CG, DG, RTCE and RTCF spaces
        on OuterProductCells.
        """
        if domain is None:
            cell = None
        else:
            domain = as_domain(domain)
            cell = domain.cell()
            ufl_assert(cell is not None, "Missing cell in given domain.")

        family, short_name, degree, value_shape, reference_value_shape, sobolev_space, mapping = \
          canonical_element_description(family, cell, degree, form_degree)

        if isinstance(cell, OuterProductCell):
            # Delay import to avoid circular dependency at module load time
            from ufl.finiteelement.outerproductelement import OuterProductElement
            from ufl.finiteelement.enrichedelement import EnrichedElement
            from ufl.finiteelement.hdivcurl import HDiv, HCurl

            # Initially degree is an integer,
            # but it is a tuple during reconstruction.
            if isinstance(degree, tuple):
                assert len(degree) == 2 and degree[0] == degree[1]
                degree = degree[0]

            if family in ["RTCF", "RTCE"]:
                ufl_assert(cell._A.topological_dimension() == 1, "%s is available on OuterProductCell(interval, interval) only." % family)
                ufl_assert(cell._B.topological_dimension() == 1, "%s is available on OuterProductCell(interval, interval) only." % family)

                C_elt = FiniteElement("CG", "interval", degree, form_degree, quad_scheme)
                D_elt = FiniteElement("DG", "interval", degree - 1, form_degree, quad_scheme)

                CxD_elt = OuterProductElement(C_elt, D_elt, domain, form_degree, quad_scheme)
                DxC_elt = OuterProductElement(D_elt, C_elt, domain, form_degree, quad_scheme)

                if family == "RTCF":
                    return EnrichedElement(HDiv(CxD_elt), HDiv(DxC_elt))
                if family == "RTCE":
                    return EnrichedElement(HCurl(CxD_elt), HCurl(DxC_elt))

            elif family in ["Lagrange", "Discontinuous Lagrange"]:
                return OuterProductElement(FiniteElement(family, cell._A, degree, form_degree, quad_scheme),
                                           FiniteElement(family, cell._B, degree, form_degree, quad_scheme),
                                           domain, form_degree, quad_scheme)

        return super(FiniteElement, cls).__new__(cls,
                                                 family,
                                                 domain,
                                                 degree,
                                                 form_degree,
                                                 quad_scheme)

    def __init__(self,
                 family,
                 domain=None,
                 degree=None,
                 form_degree=None,
                 quad_scheme=None):
        """Create finite element.

        *Arguments*
            family (string)
               The finite element family
            domain
               The geometric domain
            degree (int)
               The polynomial degree (optional)
            form_degree (int)
               The form degree (FEEC notation, used when field is
               viewed as k-form)
            quad_scheme
               The quadrature scheme (optional)
        """
        if domain is None:
            cell = None
        else:
            domain = as_domain(domain)
            cell = domain.cell()
            ufl_assert(cell is not None, "Missing cell in given domain.")

        family, short_name, degree, value_shape, reference_value_shape, sobolev_space, mapping = \
          canonical_element_description(family, cell, degree, form_degree)

        # TODO: Move these to base? Might be better to instead simplify base though.
        self._sobolev_space = sobolev_space
        self._mapping = mapping
        self._short_name = short_name

        # Finite elements on quadrilaterals have an IrreducibleInt as degree
        if domain is not None:
            if cell.cellname() == "quadrilateral":
                from ufl.algorithms.estimate_degrees import IrreducibleInt
                degree = IrreducibleInt(degree)

        # Initialize element data
        FiniteElementBase.__init__(self, family, domain, degree,
                                   quad_scheme, value_shape, reference_value_shape)

        # Cache repr string
        self._repr = "FiniteElement(%r, %r, %r, quad_scheme=%r)" % (
            self.family(), self.domain(), self.degree(), self.quadrature_scheme())
        assert '"' not in self._repr

    def mapping(self):
        return self._mapping

    def sobolev_space(self):
        return self._sobolev_space

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data, which must be reconstructed or supplied by other means.
        """
        return "FiniteElement(%r, %s, %r, %r)" % (
            self.family(), self.domain().reconstruction_signature(), self.degree(), self.quadrature_scheme())

    def signature_data(self, renumbering):
        data = ("FiniteElement", self._family, self._degree,
                self._value_shape, self._reference_value_shape,
                self._quad_scheme,
                ("no domain" if self._domain is None else self._domain.signature_data(renumbering)))
        return data

    def reconstruct(self, **kwargs):
        """Construct a new FiniteElement object with some properties
        replaced with new values."""
        kwargs["family"] = kwargs.get("family", self.family())
        kwargs["domain"] = kwargs.get("domain", self.domain())
        kwargs["degree"] = kwargs.get("degree", self.degree())
        kwargs["quad_scheme"] = kwargs.get("quad_scheme", self.quadrature_scheme())
        return FiniteElement(**kwargs)

    def __str__(self):
        "Format as string for pretty printing."
        qs = self.quadrature_scheme()
        qs = "" if qs is None else "(%s)" % qs
        return "<%s%s%s on a %s>" % (self._short_name, istr(self.degree()),\
                                           qs, self.domain())

    def shortstr(self):
        "Format as string for pretty printing."
        return "%s%s(%s)" % (self._short_name, istr(self.degree()),
                             istr(self.quadrature_scheme()))
