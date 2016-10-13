# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
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
# Modified by Massimiliano Leoni, 2016

# import six
from ufl.log import error
from ufl.utils.py23 import as_native_str
from ufl.utils.py23 import as_native_strings
from ufl.utils.formatting import istr
from ufl.cell import as_cell

from ufl.cell import TensorProductCell
from ufl.finiteelement.elementlist import canonical_element_description, simplices
from ufl.finiteelement.finiteelementbase import FiniteElementBase


# @six.python_2_unicode_compatible
class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements."
    # TODO: Move these to base?
    __slots__ = as_native_strings((
        "_short_name",
        "_sobolev_space",
        "_mapping",
        ))

    def __new__(cls,
                family,
                cell=None,
                degree=None,
                form_degree=None,
                quad_scheme=None):
        """Intercepts construction to expand CG, DG, RTCE and RTCF
        spaces on TensorProductCells."""
        if cell is not None:
            cell = as_cell(cell)

        if isinstance(cell, TensorProductCell):
            family, short_name, degree, value_shape, reference_value_shape, sobolev_space, mapping = \
                canonical_element_description(family, cell, degree, form_degree)

            # Delay import to avoid circular dependency at module load time
            from ufl.finiteelement.tensorproductelement import TensorProductElement
            from ufl.finiteelement.enrichedelement import EnrichedElement
            from ufl.finiteelement.hdivcurl import HDivElement as HDiv, HCurlElement as HCurl

            if family in ["RTCF", "RTCE"]:
                if cell._cells[0].cellname() != "interval":
                    error("%s is available on TensorProductCell(interval, interval) only." % family)
                if cell._cells[1].cellname() != "interval":
                    error("%s is available on TensorProductCell(interval, interval) only." % family)

                C_elt = FiniteElement("CG", "interval", degree, 0, quad_scheme)
                D_elt = FiniteElement("DG", "interval", degree - 1, 1, quad_scheme)

                CxD_elt = TensorProductElement(C_elt, D_elt, cell=cell)
                DxC_elt = TensorProductElement(D_elt, C_elt, cell=cell)

                if family == "RTCF":
                    return EnrichedElement(HDiv(CxD_elt), HDiv(DxC_elt))
                if family == "RTCE":
                    return EnrichedElement(HCurl(CxD_elt), HCurl(DxC_elt))

            elif family == "NCF":
                if cell._cells[0].cellname() != "quadrilateral":
                    error("%s is available on TensorProductCell(quadrilateral, interval) only." % family)
                if cell._cells[1].cellname() != "interval":
                    error("%s is available on TensorProductCell(quadrilateral, interval) only." % family)

                Qc_elt = FiniteElement("RTCF", "quadrilateral", degree, 1, quad_scheme)
                Qd_elt = FiniteElement("DQ", "quadrilateral", degree - 1, 2, quad_scheme)

                Id_elt = FiniteElement("DG", "interval", degree - 1, 1, quad_scheme)
                Ic_elt = FiniteElement("CG", "interval", degree, 0, quad_scheme)

                return EnrichedElement(HDiv(TensorProductElement(Qc_elt, Id_elt, cell=cell)),
                                       HDiv(TensorProductElement(Qd_elt, Ic_elt, cell=cell)))

            elif family == "NCE":
                if cell._cells[0].cellname() != "quadrilateral":
                    error("%s is available on TensorProductCell(quadrilateral, interval) only." % family)
                if cell._cells[1].cellname() != "interval":
                    error("%s is available on TensorProductCell(quadrilateral, interval) only." % family)

                Qc_elt = FiniteElement("Q", "quadrilateral", degree, 0, quad_scheme)
                Qd_elt = FiniteElement("RTCE", "quadrilateral", degree, 1, quad_scheme)

                Id_elt = FiniteElement("DG", "interval", degree - 1, 1, quad_scheme)
                Ic_elt = FiniteElement("CG", "interval", degree, 0, quad_scheme)

                return EnrichedElement(HCurl(TensorProductElement(Qc_elt, Id_elt, cell=cell)),
                                       HCurl(TensorProductElement(Qd_elt, Ic_elt, cell=cell)))

            elif family == "Q":
                return TensorProductElement(FiniteElement("CG", cell._cells[0], degree, 0, quad_scheme),
                                            FiniteElement("CG", cell._cells[1], degree, 0, quad_scheme),
                                            cell=cell)

            elif family == "DQ":
                family_A = "DG" if cell._cells[0].cellname() in simplices else "DQ"
                family_B = "DG" if cell._cells[1].cellname() in simplices else "DQ"
                elem_A = FiniteElement(family_A, cell._cells[0], degree,
                                       cell._cells[0].topological_dimension(), quad_scheme)
                elem_B = FiniteElement(family_B, cell._cells[1], degree,
                                       cell._cells[1].topological_dimension(), quad_scheme)
                return TensorProductElement(elem_A, elem_B, cell=cell)

        return super(FiniteElement, cls).__new__(cls)

    def __init__(self,
                 family,
                 cell=None,
                 degree=None,
                 form_degree=None,
                 quad_scheme=None):
        """Create finite element.

        *Arguments*
            family (string)
               The finite element family
            cell
               The geometric cell
            degree (int)
               The polynomial degree (optional)
            form_degree (int)
               The form degree (FEEC notation, used when field is
               viewed as k-form)
            quad_scheme
               The quadrature scheme (optional)
        """
        # Note: Unfortunately, dolfin sometimes passes None for
        # cell. Until this is fixed, allow it:
        if cell is not None:
            cell = as_cell(cell)

        family, short_name, degree, value_shape, reference_value_shape, sobolev_space, mapping = canonical_element_description(family, cell, degree, form_degree)

        # TODO: Move these to base? Might be better to instead
        # simplify base though.
        self._sobolev_space = sobolev_space
        self._mapping = mapping
        self._short_name = short_name

        # Finite elements on quadrilaterals have an IrreducibleInt as degree
        if cell is not None:
            if cell.cellname() == "quadrilateral":
                from ufl.algorithms.estimate_degrees import IrreducibleInt
                degree = IrreducibleInt(degree)

        # Initialize element data
        FiniteElementBase.__init__(self, family, cell, degree, quad_scheme,
                                   value_shape, reference_value_shape)

        # Cache repr string
        qs = self.quadrature_scheme()
        if qs is None:
            quad_str = ""
        else:
            quad_str = ", quad_scheme=%s" % repr(qs)
        self._repr = as_native_str("FiniteElement(%s, %s, %s%s)" % (
            repr(self.family()), repr(self.cell()), repr(self.degree()), quad_str))
        assert '"' not in self._repr

    def mapping(self):
        return self._mapping

    def sobolev_space(self):
        "Return the underlying Sobolev space."
        return self._sobolev_space

    def __str__(self):
        "Format as string for pretty printing."
        qs = self.quadrature_scheme()
        qs = "" if qs is None else "(%s)" % qs
        return "<%s%s%s on a %s>" % (self._short_name, istr(self.degree()),
                                     qs, self.cell())

    def shortstr(self):
        "Format as string for pretty printing."
        return "%s%s(%s)" % (self._short_name, istr(self.degree()),
                             istr(self.quadrature_scheme()))

    def __getnewargs__(self):
        """Return the arguments which pickle needs to recreate the object."""
        return (self.family(),
                self.cell(),
                self.degree(),
                None,
                self.quadrature_scheme())
