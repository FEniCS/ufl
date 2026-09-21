"""Algorithm for lowering abstractions of geometric types.

This means replacing high-level types with expressions
of mostly the Jacobian and reference cell data.
"""

# Copyright (C) 2013-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import warnings
from functools import reduce, singledispatchmethod
from itertools import combinations

import ufl.classes
from ufl.classes import (
    CellCoordinate,
    CellEdgeVectors,
    CellFacetJacobian,
    CellOrientation,
    CellOrigin,
    CellRidgeJacobian,
    CellVertices,
    CellVolume,
    Expr,
    FacetEdgeVectors,
    FacetJacobian,
    FacetJacobianDeterminant,
    FloatValue,
    Form,
    Integral,
    Jacobian,
    JacobianDeterminant,
    JacobianInverse,
    MaxCellEdgeLength,
    ReferenceCellVolume,
    ReferenceFacetVolume,
    ReferenceGrad,
    ReferenceNormal,
    RidgeJacobian,
    SpatialCoordinate,
)
from ufl.compound_expressions import cross_expr, determinant_expr, inverse_expr
from ufl.core.multiindex import Index, indices
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import extract_unique_domain
from ufl.measure import custom_integral_types, point_integral_types
from ufl.operators import conj, max_value, min_value, real, sqrt
from ufl.tensors import as_tensor, as_vector


class GeometryLoweringApplier(DAGTraverser):
    """Geometry lowering."""

    def __init__(self, preserve_types=()):
        """Initialise."""
        super().__init__()
        # Store preserve_types as boolean lookup table
        self._preserve_types = [False] * Expr._ufl_num_typecodes_
        for cls in preserve_types:
            self._preserve_types[cls._ufl_typecode_] = True

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr) -> ufl.classes.Expr:
        """Process ``o``."""
        return super().process(o)

    @process.register(ufl.classes.Expr)
    def _(self, o: ufl.classes.Expr) -> ufl.classes.Expr:
        return self.reuse_if_untouched(o)

    @process.register(ufl.classes.Terminal)
    def _(self, t: ufl.classes.Terminal) -> ufl.classes.Terminal:
        """Apply to terminal."""
        return t

    @process.register(ufl.classes.Jacobian)
    def _(self, o):
        """Apply to jacobian."""
        if self._preserve_types[o._ufl_typecode_]:
            return o
        domain = extract_unique_domain(o)
        if not domain.ufl_coordinate_element().pullback.is_identity:
            raise ValueError("Piola mapped coordinates are not implemented.")
        # Note: No longer supporting domain.coordinates(), always
        # preserving SpatialCoordinate object.  However if Jacobians
        # are not preserved, using
        # ReferenceGrad(SpatialCoordinate(domain)) to represent them.
        x = self(SpatialCoordinate(domain))
        return ReferenceGrad(x)

    def _future_jacobian(self, o):
        """Apply to _future_jacobian."""
        # If we're not using Coefficient to represent the spatial
        # coordinate, we can just as well just return o here too
        # unless we add representation of basis functions and dofs to
        # the ufl layer (which is nice to avoid).
        return o

    @process.register(ufl.classes.JacobianInverse)
    def _(self, o):
        """Apply to jacobian_inverse."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        J = self(Jacobian(domain))
        # TODO: This could in principle use
        # preserve_types[JacobianDeterminant] with minor refactoring:
        K = inverse_expr(J)
        return K

    @process.register(ufl.classes.JacobianDeterminant)
    def _(self, o):
        """Apply to jacobian_determinant."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        J = self(Jacobian(domain))
        detJ = determinant_expr(J)

        # TODO: Is "signing" the determinant for manifolds the
        #       cleanest approach?  The alternative is to have a
        #       specific type for the unsigned pseudo-determinant.
        if domain.topological_dimension < domain.geometric_dimension:
            co = CellOrientation(domain)
            detJ = co * detJ
        return detJ

    @process.register(ufl.classes.FacetJacobian)
    def _(self, o):
        """Apply to facet_jacobian."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        J = self(Jacobian(domain))
        RFJ = CellFacetJacobian(domain)
        i, j, k = indices(3)
        return as_tensor(J[i, k] * RFJ[k, j], (i, j))

    @process.register(ufl.classes.FacetJacobianInverse)
    def _(self, o):
        """Apply to facet_jacobian_inverse."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        FJ = self(FacetJacobian(domain))
        # This could in principle use
        # preserve_types[JacobianDeterminant] with minor refactoring:
        return inverse_expr(FJ)

    @process.register(ufl.classes.FacetJacobianDeterminant)
    def _(self, o):
        """Apply to facet_jacobian_determinant."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        FJ = self(FacetJacobian(domain))
        detFJ = determinant_expr(FJ)

        # TODO: Should we "sign" the facet jacobian determinant for
        #       manifolds?  It's currently used unsigned in
        #       apply_integral_scaling.
        # if domain.topological_dimension < domain.geometric_dimension:
        #     co = CellOrientation(domain)
        #     detFJ = co*detFJ

        return detFJ

    @process.register(ufl.classes.RidgeJacobian)
    def _(self, o):
        """Apply to ridge_jacobian."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        J = self(Jacobian(domain))
        REJ = CellRidgeJacobian(domain)
        i, j, k = indices(3)
        return as_tensor(J[i, k] * REJ[k, j], (i, j))

    @process.register(ufl.classes.RidgeJacobianInverse)
    def _(self, o):
        """Apply to edge_jacobian_inverse."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        EJ = self(RidgeJacobian(domain))
        return inverse_expr(EJ)

    @process.register(ufl.classes.RidgeJacobianDeterminant)
    def _(self, o):
        """Apply to edge_jacobian_determinant."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        EJ = self(RidgeJacobian(domain))
        detEJ = determinant_expr(EJ)
        return detEJ

    @process.register(ufl.classes.SpatialCoordinate)
    def _(self, o):
        """Apply to spatial_coordinate.

        Fall through to coordinate field of domain if it exists.
        """
        if self._preserve_types[o._ufl_typecode_]:
            return o
        if not extract_unique_domain(o).ufl_coordinate_element().pullback.is_identity:
            raise ValueError("Piola mapped coordinates are not implemented.")
        # No longer supporting domain.coordinates(), always preserving
        # SpatialCoordinate object.
        return o

    @process.register(ufl.classes.CellCoordinate)
    def _(self, o):
        """Apply to cell_coordinate.

        Compute from physical coordinates if they are known, using the appropriate mappings.
        """
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        K = self(JacobianInverse(domain))
        x = self(SpatialCoordinate(domain))
        x0 = CellOrigin(domain)
        i, j = indices(2)
        X = as_tensor(K[i, j] * (x[j] - x0[j]), (i,))
        return X

    @process.register(ufl.classes.CellVolume)
    def _(self, o):
        """Apply to cell_volume."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        if not domain.is_piecewise_linear_simplex_domain():
            # Don't lower for non-affine cells, instead leave it to
            # form compiler
            warnings.warn("Only know how to compute the cell volume of an affine cell.")
            return o

        r = self(JacobianDeterminant(domain))
        r0 = ReferenceCellVolume(domain)
        return abs(r * r0)

    @process.register(ufl.classes.FacetArea)
    def _(self, o):
        """Apply to facet_area."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        tdim = domain.topological_dimension
        if not domain.is_piecewise_linear_simplex_domain():
            # Don't lower for non-affine cells, instead leave it to
            # form compiler
            warnings.warn("Only know how to compute the facet area of an affine cell.")
            return o

        # Area of "facet" of interval (i.e. "area" of a vertex) is defined as 1.0
        if tdim == 1:
            return FloatValue(1.0)

        r = self(FacetJacobianDeterminant(domain))
        r0 = ReferenceFacetVolume(domain)
        return abs(r * r0)

    @process.register(ufl.classes.Circumradius)
    def _(self, o):
        """Apply to circumradius."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)

        if not domain.is_piecewise_linear_simplex_domain():
            raise ValueError("Circumradius only makes sense for affine simplex cells")

        cellname = domain.ufl_cell().cellname
        cellvolume = self(CellVolume(domain))

        if cellname == "interval":
            # Optimization for square interval; no square root needed
            return 0.5 * cellvolume

        # Compute lengths of cell edges
        edges = CellEdgeVectors(domain)
        num_edges = edges.ufl_shape[0]
        j = Index()
        elen = [real(sqrt(real(edges[e, j] * conj(edges[e, j])))) for e in range(num_edges)]

        if cellname == "triangle":
            return (elen[0] * elen[1] * elen[2]) / (4.0 * cellvolume)

        elif cellname == "tetrahedron":
            # la, lb, lc = lengths of the sides of an intermediate triangle
            # NOTE: Is here some hidden numbering assumption?
            la = elen[3] * elen[2]
            lb = elen[4] * elen[1]
            lc = elen[5] * elen[0]
            # p = perimeter
            p = la + lb + lc
            # s = semiperimeter
            s = p / 2
            # area of intermediate triangle with Herons formula
            triangle_area = sqrt(s * (s - la) * (s - lb) * (s - lc))
            return triangle_area / (6.0 * cellvolume)

    @process.register(ufl.classes.MaxCellEdgeLength)
    def _(self, o):
        """Apply to max_cell_edge_length."""
        return self._reduce_cell_edge_length(o, max_value)

    @process.register(ufl.classes.MinCellEdgeLength)
    def _(self, o):
        """Apply to min_cell_edge_length."""
        return self._reduce_cell_edge_length(o, min_value)

    def _reduce_cell_edge_length(self, o, reduction_op):
        """Apply to _reduce_cell_edge_length."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)

        if domain.ufl_coordinate_element().embedded_subdegree > 1:
            # Don't lower bendy cells, instead leave it to form compiler
            warnings.warn("Only know how to compute cell edge lengths of P1 or Q1 cell.")
            return o

        elif domain.ufl_cell().cellname == "interval":
            # Interval optimization, square root not needed
            return self(CellVolume(domain))

        else:
            # Other P1 or Q1 cells
            edges = CellEdgeVectors(domain)
            num_edges = edges.ufl_shape[0]
            j = Index()
            elen2 = [real(edges[e, j] * conj(edges[e, j])) for e in range(num_edges)]
            return real(sqrt(reduce(reduction_op, elen2)))

    @process.register(ufl.classes.CellDiameter)
    def _(self, o):
        """Apply to cell_diameter."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)

        if domain.ufl_coordinate_element().embedded_subdegree > 1:
            # Don't lower bendy cells, instead leave it to form compiler
            warnings.warn("Only know how to compute cell diameter of P1 or Q1 cell.")
            return o

        elif domain.is_piecewise_linear_simplex_domain():
            # Simplices
            return self(MaxCellEdgeLength(domain))

        else:
            # Q1 cells, maximal distance between any two vertices
            verts = CellVertices(domain)
            verts = [verts[v, ...] for v in range(verts.ufl_shape[0])]
            j = Index()
            elen2 = (real((v0 - v1)[j] * conj((v0 - v1)[j])) for v0, v1 in combinations(verts, 2))
            return real(sqrt(reduce(max_value, elen2)))

    @process.register(ufl.classes.MaxFacetEdgeLength)
    def _(self, o):
        """Apply to max_facet_edge_length."""
        return self._reduce_facet_edge_length(o, max_value)

    @process.register(ufl.classes.MinFacetEdgeLength)
    def _(self, o):
        """Apply to min_facet_edge_length."""
        return self._reduce_facet_edge_length(o, min_value)

    def _reduce_facet_edge_length(self, o, reduction_op):
        """Apply to _reduce_facet_edge_length."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)

        if domain.ufl_cell().topological_dimension < 3:
            raise ValueError("Facet edge lengths only make sense for topological dimension >= 3.")

        elif domain.ufl_coordinate_element().embedded_subdegree > 1:
            # Don't lower bendy cells, instead leave it to form compiler
            warnings.warn("Only know how to compute facet edge lengths of P1 or Q1 cell.")
            return o

        else:
            # P1 tetrahedron or Q1 hexahedron
            edges = FacetEdgeVectors(domain)
            num_edges = edges.ufl_shape[0]
            j = Index()
            elen2 = [real(edges[e, j] * conj(edges[e, j])) for e in range(num_edges)]
            return real(sqrt(reduce(reduction_op, elen2)))

    @process.register(ufl.classes.CellNormal)
    def _(self, o):
        """Apply to cell_normal."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        gdim = domain.geometric_dimension
        tdim = domain.topological_dimension

        if tdim == gdim - 1:  # n-manifold embedded in n-1 space
            i = Index()
            J = self(Jacobian(domain))

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
                raise ValueError(f"Cell normal not implemented for tdim {tdim}, gdim {gdim}")

            # Return normalized vector, sign corrected by cell
            # orientation
            co = CellOrientation(domain)
            return co * cell_normal / sqrt(cell_normal[i] * cell_normal[i])
        else:
            raise ValueError(f"Cell normal undefined for tdim {tdim}, gdim {gdim}")

    @process.register(ufl.classes.FacetNormal)
    def _(self, o):
        """Apply to facet_normal."""
        if self._preserve_types[o._ufl_typecode_]:
            return o

        domain = extract_unique_domain(o)
        tdim = domain.topological_dimension

        if tdim == 1:
            # Special-case 1D (possibly immersed), for which we say
            # that n is just in the direction of J.
            J = self(Jacobian(domain))  # dx/dX
            ndir = J[:, 0]

            gdim = domain.geometric_dimension
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
            Jinv = self(JacobianInverse(domain))
            i, j = indices(2)

            rn = ReferenceNormal(domain)
            # compute signed, unnormalised normal; note transpose
            ndir = as_vector(Jinv[j, i] * rn[j], i)

            # normalise
            i = Index()
            n = ndir / sqrt(ndir[i] * ndir[i])
            r = n

        if r.ufl_shape != o.ufl_shape:
            raise ValueError(
                f"Inconsistent dimensions (in={o.ufl_shape[0]}, out={r.ufl_shape[0]})."
            )
        return r


def apply_geometry_lowering(form, preserve_types=()):
    """Change GeometricQuantity objects in expression to the lowest level GeometricQuantity objects.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    Args:
        form: An Expr or Form.
        preserve_types: Preserved types
    """
    if isinstance(form, Form):
        newintegrals = [
            apply_geometry_lowering(integral, preserve_types) for integral in form.integrals()
        ]
        return Form(newintegrals)

    elif isinstance(form, Integral):
        integral = form
        if integral.integral_type() in (custom_integral_types + point_integral_types):
            automatic_preserve_types = [SpatialCoordinate, Jacobian]
        else:
            automatic_preserve_types = [CellCoordinate]
        preserve_types = set(preserve_types) | set(automatic_preserve_types)

        mf = GeometryLoweringApplier(preserve_types)
        newintegrand = mf(integral.integrand())
        return integral.reconstruct(integrand=newintegrand)

    elif isinstance(form, Expr):
        expr = form
        mf = GeometryLoweringApplier(preserve_types)
        return mf(expr)

    else:
        raise ValueError(f"Invalid type {form.__class__.__name__}")
