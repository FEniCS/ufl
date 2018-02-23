# -*- coding: utf-8 -*-
"""Algorithm for Jacobian rewriting.

Needed in case of several meshes 
(could meshes with different dimension)
"""

# Copyright (C) 2013-2016 Martin Sandve Alnæs
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

from six.moves import xrange as range

from ufl.log import error, warning

from ufl.core.multiindex import Index, indices
from ufl.corealg.multifunction import MultiFunction, memoized_handler
from ufl.corealg.map_dag import map_expr_dag
from ufl.measure import custom_integral_types, point_integral_types

from ufl.classes import (Expr, Form, Integral,
                         ReferenceGrad,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         CellOrientation, CellOrigin, CellCoordinate,
                         FacetJacobian, FacetJacobianDeterminant,
                         CellFacetJacobian,
                         CellEdgeVectors, FacetEdgeVectors,
                         ReferenceNormal,
                         ReferenceCellVolume, ReferenceFacetVolume,
                         CellVolume, FacetArea,
                         SpatialCoordinate,
                         FloatValue)
# FacetJacobianInverse,
# FacetOrientation, QuadratureWeight,

from ufl.tensors import as_tensor, as_vector
from ufl.operators import sqrt, max_value, min_value

from ufl.compound_expressions import determinant_expr, cross_expr, inverse_expr
from ufl.algorithms.apply_geometry_lowering import GeometryLoweringApplier

class JacobianRewritingApplier(MultiFunction):
    def __init__(self, preserve_types=()):
        MultiFunction.__init__(self)
        # Store preserve_types as boolean lookup table
        self._preserve_types = [False]*Expr._ufl_num_typecodes_
        for cls in preserve_types:
            self._preserve_types[cls._ufl_typecode_] = True

        self._geometry_lowering_applier = GeometryLoweringApplier(preserve_types)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t
    
    @memoized_handler
    def jacobian(self, o):
        domain = o.ufl_domain()
        # Argument domain - geo/topo dimension
        tdim= domain.topological_dimension()
        gdim= domain.geometric_dimension()

        ## Compare the topo dim
        if tdim == gdim:
            # Reuse the Jacobian of the integration domain
            return self._geometry_lowering_applier.jacobian(o)
        elif tdim == gdim - 1 : ## 2D-1D or 3D-2D
            return self._geometry_lowering_applier.facet_jacobian(o)
        elif tdim == gdim - 2: ## 3D-1D (not yet implemented)
            print("3D-1D case - not yet implemented")
            # TODO? : EdgeJacobian = Jacobian(ref cell to real cell) * CellFacetJacobian(ref facet to ref cell) * CellEdgeJacobian(ref interval to ref facet)
            # with CellEdgeJacobian(domain) = CellFacetJacobian(any 2D domain)
            return o

    @memoized_handler
    def jacobian_determinant(self, o):
        domain = o.ufl_domain()
        
        J = self.jacobian(Jacobian(domain))
        detJ = determinant_expr(J)

        if domain.topological_dimension() < domain.geometric_dimension():
            co = CellOrientation(domain)
            detJ = co*detJ

        return detJ
    
def apply_jacobian_rewriting(form, preserve_types=()):
    """Rewrites the Jacobian introduced through GeometryLoweringApplier.

    @param form:
        An Expr or Form.
    """
    if isinstance(form, Form):
        newintegrals = [apply_jacobian_rewriting(integral, preserve_types)
                        for integral in form.integrals()]
        return Form(newintegrals)

    elif isinstance(form, Integral):
        integral = form
        if integral.integral_type() in (custom_integral_types + point_integral_types):
            automatic_preserve_types = [SpatialCoordinate, Jacobian]
        else:
            automatic_preserve_types = [CellCoordinate]
        preserve_types = set(preserve_types) | set(automatic_preserve_types)

        mf = JacobianRewritingApplier(preserve_types)
        newintegrand = map_expr_dag(mf, integral.integrand())
        return integral.reconstruct(integrand=newintegrand)

    elif isinstance(form, Expr):
        expr = form
        mf = JacobianRewritingApplier(preserve_types)
        return map_expr_dag(mf, expr)

    else:
        error("Invalid type %s" % (form.__class__.__name__,))
