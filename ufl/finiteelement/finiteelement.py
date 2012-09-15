"This module defines the UFL finite element classes."

# Copyright (C) 2008-2012 Martin Sandve Alnes
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
#
# First added:  2008-03-03
# Last changed: 2012-08-16

from itertools import izip
from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, index_to_component, component_to_index, istr, EmptyDict
from ufl.geometry import as_cell, domain2facet, ProductCell
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.elementlist import ufl_elements, aliases
from ufl.finiteelement.finiteelementbase import FiniteElementBase

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"
    def __init__(self, family, cell=None, degree=None, quad_scheme=None,
                 form_degree=None):
        """Create finite element

        *Arguments*
            family (string)
               The finite element family
            cell
               The cell
            degree (int)
               The polynomial degree (optional)
            quad_scheme
               The quadrature scheme (optional)
            form_degree (int)
               The form degree (FEEC notation, used when field is
               viewed as k-form)
        """

        # Map evt. string argument to a Cell
        cell = as_cell(cell)

        # Check whether this family is an alias for something else
        if family in aliases:
            (name, cell, r) = aliases[family](family, cell, degree, form_degree)
            #info_blue("%s, is an alias for %s " % (
            #        (family, cell, degree, form_degree),
            #        (name, cell, r)))
            self.__init__(name, cell, r, quad_scheme)
            return

        # Check that the element family exists
        ufl_assert(family in ufl_elements,
                   'Unknown finite element "%s".' % family)

        # Check that element data is valid (and also get common family name)
        (family, self._short_name, value_rank, krange, domains) =\
            ufl_elements[family]

        # Validate domain if a valid cell is specified
        if cell.is_undefined():
            # Case of invalid cell, some stuff is then undefined,
            # such as the domain and some dimensions
            pass
        else:
            domain = cell.domain()
            ufl_assert(domain in domains,
                       'Domain "%s" invalid for "%s" finite element.' % (domain, family))

        # Validate degree if specified
        if degree is not None:
            ufl_assert(krange is not None,
                       'Degree "%s" invalid for "%s" finite element, '\
                           'should be None.' % (degree, family))
            kmin, kmax = krange
            ufl_assert(kmin is None or degree >= kmin,
                       'Degree "%s" invalid for "%s" finite element.' %\
                           (degree, family))
            ufl_assert(kmax is None or degree <= kmax,
                   'Degree "%s" invalid for "%s" finite element.' %\
                           (istr(degree), family))

        # Set value dimension (default to using domain dimension in each axis)
        if value_rank == 0:
            value_shape = ()
        else:
            ufl_assert(not cell.is_undefined(),
                       "Cannot infer value shape with an undefined cell.")
            dim = cell.geometric_dimension()
            value_shape = (dim,)*value_rank

        # Initialize element data
        super(FiniteElement, self).__init__(family, cell, degree,
                                            quad_scheme, value_shape)

        # Cache repr string
        self._repr = "FiniteElement(%r, %r, %r, %r)" % (
            self.family(), self.cell(), self.degree(), self.quadrature_scheme())

    def reconstruct(self, **kwargs):
        """Construct a new FiniteElement object with some properties
        replaced with new values."""
        kwargs["family"] = kwargs.get("family", self.family())
        kwargs["cell"] = kwargs.get("cell", self.cell())
        kwargs["degree"] = kwargs.get("degree", self.degree())
        kwargs["quad_scheme"] = kwargs.get("quad_scheme", self.quadrature_scheme())
        return FiniteElement(**kwargs)

    def __str__(self):
        "Format as string for pretty printing."
        qs = self.quadrature_scheme()
        qs = "" if qs is None else "(%s)" % qs
        return "<%s%s%s on a %s>" % (self._short_name, istr(self.degree()),\
                                           qs, self.cell())

    def shortstr(self):
        "Format as string for pretty printing."
        return "%s%s(%s)" % (self._short_name, istr(self.degree()), istr(self.quadrature_scheme()))
