# -*- coding: utf-8 -*-
"""Algorithm for lowering abstractions of geometric types.

This means replacing high-level types with expressions
of mostly the Jacobian and reference cell data.
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

from functools import reduce
from itertools import combinations

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
                         MaxCellEdgeLength,
                         CellEdgeVectors, FacetEdgeVectors, CellVertices,
                         ReferenceNormal,
                         ReferenceCellVolume, ReferenceFacetVolume, CellVolume,
                         SpatialCoordinate,
                         FloatValue)
# FacetJacobianInverse,
# FacetOrientation, QuadratureWeight,

from ufl.tensors import as_tensor, as_vector
from ufl.operators import sqrt, max_value, min_value

from ufl.compound_expressions import determinant_expr, cross_expr, inverse_expr


class GeometryLoweringApplier(MultiFunction):
    def __init__(self, preserve_types=()):
        MultiFunction.__init__(self)
        # Store preserve_types as boolean lookup table
        self._preserve_types = [False] * Expr._ufl_num_typecodes_
        for cls in preserve_types:
            self._preserve_types[cls._ufl_typecode_] = True

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    @memoized_handler
    def jacobian(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o
        domain = o.ufl_domain()
        if domain.ufl_coordinate_element().mapping() != "identity":
            error("Piola mapped coordinates are not implemented.")
        # Note: No longer supporting domain.coordinates(), always
        # preserving SpatialCoordinate object.  However if Jacobians
        # are not preserved, using
        # ReferenceGrad(SpatialCoordinate(domain)) to represent them.
        x = self.spatial_coordinate(SpatialCoordinate(domain))
        return ReferenceGrad(x)

    @memoized_handler
    def _future_jacobian(self, o):
        # If we're not using Coefficient to represent the spatial
        # coordinate, we can just as well just return o here too
        # unless we add representation of basis functions and dofs to
        # the ufl layer (which is nice to avoid).
        return o

    @memoized_handler
    def jacobian_inverse(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        J = self.jacobian(Jacobian(domain))
        # TODO: This could in principle use
        # preserve_types[JacobianDeterminant] with minor refactoring:
        K = inverse_expr(J)
        return K

    @memoized_handler
    def jacobian_determinant(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        J = self.jacobian(Jacobian(domain))
        detJ = determinant_expr(J)

        # TODO: Is "signing" the determinant for manifolds the
        #       cleanest approach?  The alternative is to have a
        #       specific type for the unsigned pseudo-determinant.
        if domain.topological_dimension() < domain.geometric_dimension():
            co = CellOrientation(domain)
            detJ = co * detJ

        return detJ

    @memoized_handler
    def facet_jacobian(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        J = self.jacobian(Jacobian(domain))
        RFJ = CellFacetJacobian(domain)
        i, j, k = indices(3)
        return as_tensor(J[i, k] * RFJ[k, j], (i, j))

    @memoized_handler
    def facet_jacobian_inverse(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        FJ = self.facet_jacobian(FacetJacobian(domain))
        # This could in principle use
        # preserve_types[JacobianDeterminant] with minor refactoring:
        return inverse_expr(FJ)

    @memoized_handler
    def facet_jacobian_determinant(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        FJ = self.facet_jacobian(FacetJacobian(domain))
        detFJ = determinant_expr(FJ)

        # TODO: Should we "sign" the facet jacobian determinant for
        #       manifolds?  It's currently used unsigned in
        #       apply_integral_scaling.
        # if domain.topological_dimension() < domain.geometric_dimension():
        #     co = CellOrientation(domain)
        #     detFJ = co*detFJ

        return detFJ

    @memoized_handler
    def spatial_coordinate(self, o):
        "Fall through to coordinate field of domain if it exists."
        if self._preserve_types[o._ufl_typecode_]:
            return o
        if o.ufl_domain().ufl_coordinate_element().mapping() != "identity":
            error("Piola mapped coordinates are not implemented.")
        # No longer supporting domain.coordinates(), always preserving
        # SpatialCoordinate object.
        return o

    @memoized_handler
    def cell_coordinate(self, o):
        "Compute from physical coordinates if they are known, using the appropriate mappings."
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        K = self.jacobian_inverse(JacobianInverse(domain))
        x = self.spatial_coordinate(SpatialCoordinate(domain))
        x0 = CellOrigin(domain)
        i, j = indices(2)
        X = as_tensor(K[i, j] * (x[j] - x0[j]), (i,))
        return X

    @memoized_handler
    def facet_cell_coordinate(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        error("Missing computation of facet reference coordinates "
              "from physical coordinates via mappings.")

    @memoized_handler
    def cell_volume(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        if not domain.is_piecewise_linear_simplex_domain():
            # Don't lower for non-affine cells, instead leave it to
            # form compiler
            warning("Only know how to compute the cell volume of an affine cell.")
            return o

        r = self.jacobian_determinant(JacobianDeterminant(domain))
        r0 = ReferenceCellVolume(domain)
        return abs(r * r0)

    @memoized_handler
    def facet_area(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        tdim = domain.topological_dimension()
        if not domain.is_piecewise_linear_simplex_domain():
            # Don't lower for non-affine cells, instead leave it to
            # form compiler
            warning("Only know how to compute the facet area of an affine cell.")
            return o

        # Area of "facet" of interval (i.e. "area" of a vertex) is defined as 1.0
        if tdim == 1:
            return FloatValue(1.0)

        r = self.facet_jacobian_determinant(FacetJacobianDeterminant(domain))
        r0 = ReferenceFacetVolume(domain)
        return abs(r * r0)

    @memoized_handler
    def circumradius(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()

        if not domain.is_piecewise_linear_simplex_domain():
            error("Circumradius only makes sense for affine simplex cells")

        cellname = domain.ufl_cell().cellname()
        cellvolume = self.cell_volume(CellVolume(domain))

        if cellname == "interval":
            # Optimization for square interval; no square root needed
            return 0.5 * cellvolume

        # Compute lengths of cell edges
        edges = CellEdgeVectors(domain)
        num_edges = edges.ufl_shape[0]
        j = Index()
        elen = [sqrt(edges[e, j] * edges[e, j]) for e in range(num_edges)]

        if cellname == "triangle":
            return (elen[0] * elen[1] * elen[2]) / (4.0 * cellvolume)

        elif cellname == "tetrahedron":
            # la, lb, lc = lengths of the sides of an intermediate triangle
            # NOTE: Is here some hidden numbering assumption?
            la = elen[3] * elen[2]
            lb = elen[4] * elen[1]
            lc = elen[5] * elen[0]
            # p = perimeter
            p = (la + lb + lc)
            # s = semiperimeter
            s = p / 2
            # area of intermediate triangle with Herons formula
            triangle_area = sqrt(s * (s - la) * (s - lb) * (s - lc))
            return triangle_area / (6.0 * cellvolume)

    @memoized_handler
    def max_cell_edge_length(self, o):
        return self._reduce_cell_edge_length(o, max_value)

    @memoized_handler
    def min_cell_edge_length(self, o):
        return self._reduce_cell_edge_length(o, min_value)

    def _reduce_cell_edge_length(self, o, reduction_op):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()

        if not domain.ufl_coordinate_element().degree() == 1:
            # Don't lower bendy cells, instead leave it to form compiler
            warning("Only know how to compute cell edge lengths of P1 or Q1 cell.")
            return o

        elif domain.ufl_cell().cellname() == "interval":
            # Interval optimization, square root not needed
            return self.cell_volume(CellVolume(domain))

        else:
            # Other P1 or Q1 cells
            edges = CellEdgeVectors(domain)
            num_edges = edges.ufl_shape[0]
            j = Index()
            elen2 = [edges[e, j] * edges[e, j] for e in range(num_edges)]
            return sqrt(reduce(reduction_op, elen2))

    @memoized_handler
    def cell_diameter(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()

        if not domain.ufl_coordinate_element().degree() == 1:
            # Don't lower bendy cells, instead leave it to form compiler
            warning("Only know how to compute cell diameter of P1 or Q1 cell.")
            return o

        elif domain.is_piecewise_linear_simplex_domain():
            # Simplices
            return self.max_cell_edge_length(MaxCellEdgeLength(domain))

        else:
            # Q1 cells, maximal distance between any two vertices
            verts = CellVertices(domain)
            verts = [verts[v, ...] for v in range(verts.ufl_shape[0])]
            j = Index()
            elen2 = ((v0 - v1)[j] * (v0 - v1)[j] for v0, v1 in combinations(verts, 2))
            return sqrt(reduce(max_value, elen2))

    @memoized_handler
    def max_facet_edge_length(self, o):
        return self._reduce_facet_edge_length(o, max_value)

    @memoized_handler
    def min_facet_edge_length(self, o):
        return self._reduce_facet_edge_length(o, min_value)

    def _reduce_facet_edge_length(self, o, reduction_op):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()

        if domain.ufl_cell().topological_dimension() < 3:
            error("Facet edge lengths only make sense for topological dimension >= 3.")

        elif not domain.ufl_coordinate_element().degree() == 1:
            # Don't lower bendy cells, instead leave it to form compiler
            warning("Only know how to compute facet edge lengths of P1 or Q1 cell.")
            return o

        else:
            # P1 tetrahedron or Q1 hexahedron
            edges = FacetEdgeVectors(domain)
            num_edges = edges.ufl_shape[0]
            j = Index()
            elen2 = [edges[e, j] * edges[e, j] for e in range(num_edges)]
            return sqrt(reduce(reduction_op, elen2))

    @memoized_handler
    def cell_normal(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        gdim = domain.geometric_dimension()
        tdim = domain.topological_dimension()

        if tdim == gdim - 1:  # n-manifold embedded in n-1 space
            i = Index()
            J = self.jacobian(Jacobian(domain))

            if tdim == 2:
                # Surface in 3D
                t0 = as_vector(J[i, 0], i)
                t1 = as_vector(J[i, 1], i)
                cell_normal = cross_expr(t0, t1)
            elif tdim == 1:
                # Line in 2D (cell normal is 'up' for a line pointing
                # to the 'right')
                cell_normal = as_vector((-J[1, 0], J[0, 0]))
            else:
                error("Cell normal not implemented for tdim %d, gdim %d" % (tdim, gdim))

            # Return normalized vector, sign corrected by cell
            # orientation
            co = CellOrientation(domain)
            return co * cell_normal / sqrt(cell_normal[i] * cell_normal[i])
        else:
            error("What do you want cell normal in gdim={0}, tdim={1} to be?".format(gdim, tdim))

    @memoized_handler
    def facet_normal(self, o):
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = o.ufl_domain()
        tdim = domain.topological_dimension()

        if tdim == 1:
            # Special-case 1D (possibly immersed), for which we say
            # that n is just in the direction of J.
            J = self.jacobian(Jacobian(domain))  # dx/dX
            ndir = J[:, 0]

            gdim = domain.geometric_dimension()
            if gdim == 1:
                nlen = abs(ndir[0])
            else:
                i = Index()
                nlen = sqrt(ndir[i] * ndir[i])

            rn = ReferenceNormal(domain)  # +/- 1.0 here
            n = rn[0] * ndir / nlen
            r = n
        else:
            # Recall that the covariant Piola transform u -> J^(-T)*u
            # preserves tangential components. The normal vector is
            # characterised by having zero tangential component in
            # reference and physical space.
            Jinv = self.jacobian_inverse(JacobianInverse(domain))
            i, j = indices(2)

            rn = ReferenceNormal(domain)
            # compute signed, unnormalised normal; note transpose
            ndir = as_vector(Jinv[j, i] * rn[j], i)

            # normalise
            i = Index()
            n = ndir / sqrt(ndir[i] * ndir[i])
            r = n

        if r.ufl_shape != o.ufl_shape:
            error("Inconsistent dimensions (in=%d, out=%d)." % (o.ufl_shape[0], r.ufl_shape[0]))
        return r


def apply_geometry_lowering(form, preserve_types=()):
    """Change GeometricQuantity objects in expression to the lowest level GeometricQuantity objects.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param form:
        An Expr or Form.
    """
    if isinstance(form, Form):
        newintegrals = [apply_geometry_lowering(integral, preserve_types)
                        for integral in form.integrals()]
        return Form(newintegrals)

    elif isinstance(form, Integral):
        integral = form
        if integral.integral_type() in (custom_integral_types + point_integral_types):
            automatic_preserve_types = [SpatialCoordinate, Jacobian]
        else:
            automatic_preserve_types = [CellCoordinate]
        preserve_types = set(preserve_types) | set(automatic_preserve_types)

        mf = GeometryLoweringApplier(preserve_types)
        newintegrand = map_expr_dag(mf, integral.integrand())
        return integral.reconstruct(integrand=newintegrand)

    elif isinstance(form, Expr):
        expr = form
        mf = GeometryLoweringApplier(preserve_types)
        return map_expr_dag(mf, expr)

    else:
        error("Invalid type %s" % (form.__class__.__name__,))
