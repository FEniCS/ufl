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

from ufl.finiteelement.elementlist import canonical_element_description
from ufl.finiteelement.finiteelementbase import FiniteElementBase

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"
    # TODO: Move these to base?
    __slots__ = ("_short_name",
                 "_sobolev_space",
                 "_mapping",
                )
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

        # Initialize element data
        FiniteElementBase.__init__(self, family, domain, degree,
                                   quad_scheme, value_shape, reference_value_shape)

        # Cache repr string
        self._repr = "FiniteElement(%r, %r, %r, quad_scheme=%r)" % (
            self.family(), self.domain(), self.degree(), self.quadrature_scheme())
        assert '"' not in self._repr

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
