"Algorithms related to restrictions."

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
# First added:  2009-05-14
# Last changed: 2012-04-12

from ufl.expr import Expr
from ufl.classes import Measure
from ufl.assertions import ufl_assert
from ufl.algorithms.transformer import Transformer, ReuseTransformer, apply_transformer

class RestrictionPropagator(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)
        self.current_restriction = None

    def restricted(self, o):
        ufl_assert(self.current_restriction is None,
            "Not expecting twice restricted expression.")
        self.current_restriction = o._side
        e, = o.operands()
        r = self.visit(e)
        self.current_restriction = None
        return r

    def facet_normal(self, o):
        ufl_assert(self.current_restriction is not None, "FacetNormal must be restricted.")
        return o(self.current_restriction)

    def jacobian(self, o):
        ufl_assert(self.current_restriction is not None, "Jacobian must be restricted.")
        return o(self.current_restriction)

    def jacobian_determinant(self, o):
        ufl_assert(self.current_restriction is not None, "JacobianDeterminant must be restricted.")
        return o(self.current_restriction)

    def jacobian_inverse(self, o):
        ufl_assert(self.current_restriction is not None, "JacobianInverse must be restricted.")
        return o(self.current_restriction)

    def cell_volume(self, o):
        ufl_assert(self.current_restriction is not None, "CellVolume must be restricted.")
        return o(self.current_restriction)

    def circumradius(self, o):
        ufl_assert(self.current_restriction is not None, "Circumradius must be restricted.")
        return o(self.current_restriction)

    # Not implemented anywhere
    #def cell_normal(self, o):
    #    ufl_assert(self.current_restriction is not None, "CellNormal must be restricted.")
    #    return o(self.current_restriction)

    # Not implemented anywhere
    #def cell_barycenter(self, o):
    #    ufl_assert(self.current_restriction is not None, "CellBarycenter must be restricted.")
    #    return o(self.current_restriction)

    # Not implemented anywhere
    #def cell_surface_area(self, o):
    #    ufl_assert(self.current_restriction is not None, "CellSurfaceArea must be restricted.")
    #    return o(self.current_restriction)

    # These are all the same from both sides of a facet:
    # facet_area, min_facet_edge_length, max_facet_edge_length,
    # facet_jacobian, facet_jacobian_determinant, facet_jacobian_inverse,
    # These as well, but also not implemented anywhere:
    # facet_barycenter, facet_diameter

    def form_argument(self, o):
        ufl_assert(self.current_restriction is not None, "Form argument must be restricted.")
        #if self.current_restriction is None:
        #    return o
        return o(self.current_restriction)

    def variable(self, o):
        ufl_assert(self.current_restriction is not None, "Form argument must be restricted.")
        #if self.current_restriction is None:
        #    return o
        return o(self.current_restriction)

class RestrictionChecker(Transformer):
    def __init__(self, require_restriction):
        Transformer.__init__(self)
        self.current_restriction = None
        self.require_restriction = require_restriction

    def expr(self, o):
        pass

    def restricted(self, o):
        ufl_assert(self.current_restriction is None,
            "Not expecting twice restricted expression.")
        self.current_restriction = o._side
        e, = o.operands()
        self.visit(e)
        self.current_restriction = None

    def facet_normal(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None, "Facet normal must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None, "Restrictions are only allowed for interior facet integrals.")

    def form_argument(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None, "Form argument must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None, "Restrictions are only allowed for interior facet integrals.")

def propagate_restrictions(expression):
    "Propagate restriction nodes to wrap terminal objects directly."
    return apply_transformer(expression, RestrictionPropagator(), integral_type=[Measure.INTERIOR_FACET, Measure.INTERIOR_FACET_HORIZ, Measure.INTERIOR_FACET_VERT])

def check_restrictions(expression, require_restriction):
    ufl_assert(isinstance(expression, Expr), "Expecting Expr instance.")
    return RestrictionChecker(require_restriction).visit(expression)
