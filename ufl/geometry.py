"""Types for representing symbolic expressions for geometric quantities."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.terminal import Terminal
from ufl.core.ufl_type import ufl_type
from ufl.domain import as_domain, extract_unique_domain

"""

Possible coordinate bootstrapping:

Xf = Xf[q]
    FacetCoordinate = quadrature point on facet (ds,dS)

X = X[q]
    CellCoordinate = quadrature point on cell (dx)

x = x[q]
    SpatialCoordinate = quadrature point from input array (dc)


Jacobians of mappings between coordinates:

Jcf = dX/dXf = grad_Xf X(Xf)
    CellFacetJacobian

Jxc = dx/dX = grad_X x(X)
    Jacobian

Jxf = dx/dXf = grad_Xf x(Xf)  =  Jxc Jcf = dx/dX dX/dXf = grad_X x(X) grad_Xf X(Xf)
    FacetJacobian = Jacobian * CellFacetJacobian


Possible computation of X from Xf:

X = Jcf Xf + X0f
    CellCoordinate = CellFacetJacobian * FacetCoordinate + CellFacetOrigin


Possible computation of x from X:

x = f(X)
    SpatialCoordinate = sum_k xdofs_k * xphi_k(X)

x = Jxc X + x0
    SpatialCoordinate = Jacobian * CellCoordinate + CellOrigin


Possible computation of x from Xf:

x = x(X(Xf))

x = Jxf Xf + x0f
    SpatialCoordinate = FacetJacobian * FacetCoordinate + FacetOrigin


Inverse relations:

X = K * (x - x0)
    CellCoordinate = JacobianInverse * (SpatialCoordinate - CellOrigin)

Xf = FK * (x - x0f)
    FacetCoordinate = FacetJacobianInverse * (SpatialCoordinate - FacetOrigin)

Xf = CFK * (X - X0f)
    FacetCoordinate = CellFacetJacobianInverse * (CellCoordinate - CellFacetOrigin)

"""


# --- Expression node types

@ufl_type(is_abstract=True)
class GeometricQuantity(Terminal):
    __slots__ = ("_domain",)

    def __init__(self, domain):
        Terminal.__init__(self)
        self._domain = as_domain(domain)

    def ufl_domains(self):
        return (self._domain,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell (or over each facet for facet quantities)."
        # NB! Geometric quantities are piecewise constant by
        # default. Override if needed.
        return True

    # NB! Geometric quantities are scalar by default. Override if
    # needed.
    ufl_shape = ()

    def _ufl_signature_data_(self, renumbering):
        "Signature data of geometric quantities depend on the domain numbering."
        return (self._ufl_class_.__name__,) + self._domain._ufl_signature_data_(renumbering)

    def __str__(self):
        return self._ufl_class_.name

    def __repr__(self):
        r = "%s(%s)" % (self._ufl_class_.__name__, repr(self._domain))
        return r

    def _ufl_compute_hash_(self):
        return hash((type(self).__name__,) + self._domain._ufl_hash_data_())

    def __eq__(self, other):
        return isinstance(other, self._ufl_class_) and other._domain == self._domain


@ufl_type(is_abstract=True)
class GeometricCellQuantity(GeometricQuantity):
    __slots__ = ()


@ufl_type(is_abstract=True)
class GeometricFacetQuantity(GeometricQuantity):
    __slots__ = ()


# --- Coordinate represented in different coordinate systems

@ufl_type()
class SpatialCoordinate(GeometricCellQuantity):
    """UFL geometry representation: The coordinate in a domain.

    In the context of expression integration,
    represents the domain coordinate of each quadrature point.

    In the context of expression evaluation in a point,
    represents the value of that point.
    """
    __slots__ = ()
    name = "x"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension
        of the domain)."""
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only case this is true is if the domain is a vertex cell.
        t = self._domain.topological_dimension()
        return t == 0

    def evaluate(self, x, mapping, component, index_values):
        "Return the value of the coordinate."
        if component == ():
            if isinstance(x, (tuple, list)):
                return float(x[0])
            else:
                return float(x)
        else:
            return float(x[component[0]])

    def count(self):
        # FIXME: Hack to make SpatialCoordinate behave like a coefficient.
        # When calling `derivative`, the count is used to sort over.
        return -1


@ufl_type()
class CellCoordinate(GeometricCellQuantity):
    """UFL geometry representation: The coordinate in a reference cell.

    In the context of expression integration,
    represents the reference cell coordinate of each quadrature point.

    In the context of expression evaluation in a point in a cell,
    represents that point in the reference coordinate system of the cell.
    """
    __slots__ = ()
    name = "X"

    @property
    def ufl_shape(self):
        t = self._domain.topological_dimension()
        return (t,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only case this is true is if the domain is a vertex cell.
        t = self._domain.topological_dimension()
        return t == 0


@ufl_type()
class FacetCoordinate(GeometricFacetQuantity):
    """UFL geometry representation: The coordinate in a reference cell of a facet.

    In the context of expression integration over a facet,
    represents the reference facet coordinate of each quadrature point.

    In the context of expression evaluation in a point on a facet,
    represents that point in the reference coordinate system of the facet.
    """
    __slots__ = ()
    name = "Xf"

    def __init__(self, domain):
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("FacetCoordinate is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        t = self._domain.topological_dimension()
        return (t - 1,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only case this is true is if the domain is an interval cell
        # (with a vertex facet).
        t = self._domain.topological_dimension()
        return t <= 1


# --- Origin of coordinate systems in larger coordinate systems

@ufl_type()
class CellOrigin(GeometricCellQuantity):
    """UFL geometry representation: The spatial coordinate corresponding to origin of a reference cell."""
    __slots__ = ()
    name = "x0"

    @property
    def ufl_shape(self):
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        return True


@ufl_type()
class FacetOrigin(GeometricFacetQuantity):
    """UFL geometry representation: The spatial coordinate corresponding to origin of a reference facet."""
    __slots__ = ()
    name = "x0f"

    @property
    def ufl_shape(self):
        g = self._domain.geometric_dimension()
        return (g,)


@ufl_type()
class CellFacetOrigin(GeometricFacetQuantity):
    """UFL geometry representation: The reference cell coordinate corresponding to origin of a reference facet."""
    __slots__ = ()
    name = "X0f"

    @property
    def ufl_shape(self):
        t = self._domain.topological_dimension()
        return (t,)


# --- Jacobians of mappings between coordinate systems

@ufl_type()
class Jacobian(GeometricCellQuantity):
    """UFL geometry representation: The Jacobian of the mapping from reference cell to spatial coordinates.

    .. math:: J_{ij} = \\frac{dx_i}{dX_j}
    """
    __slots__ = ()
    name = "J"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension
        of the domain)."""
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (g, t)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


@ufl_type()
class FacetJacobian(GeometricFacetQuantity):
    """UFL geometry representation: The Jacobian of the mapping from reference facet to spatial coordinates.

      FJ_ij = dx_i/dXf_j

    The FacetJacobian is the product of the Jacobian and CellFacetJacobian:

      FJ = dx/dXf = dx/dX dX/dXf = J * CFJ

    """
    __slots__ = ()
    name = "FJ"

    def __init__(self, domain):
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("FacetJacobian is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (g, t - 1)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


@ufl_type()
class CellFacetJacobian(GeometricFacetQuantity):  # dX/dXf
    """UFL geometry representation: The Jacobian of the mapping from reference facet to reference cell coordinates.

    CFJ_ij = dX_i/dXf_j
    """
    __slots__ = ()
    name = "CFJ"

    def __init__(self, domain):
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellFacetJacobian is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        t = self._domain.topological_dimension()
        return (t, t - 1)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always a constant mapping between two reference
        # coordinate systems.
        return True


@ufl_type()
class ReferenceCellEdgeVectors(GeometricCellQuantity):
    """UFL geometry representation: The vectors between reference cell vertices for each edge in cell."""
    __slots__ = ()
    name = "RCEV"

    def __init__(self, domain):
        GeometricCellQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellEdgeVectors is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        cell = extract_unique_domain(self).ufl_cell()
        ne = cell.num_edges()
        t = cell.topological_dimension()
        return (ne, t)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always constant for a given cell type
        return True


@ufl_type()
class ReferenceFacetEdgeVectors(GeometricFacetQuantity):
    """UFL geometry representation: The vectors between reference cell vertices for each edge in current facet."""
    __slots__ = ()
    name = "RFEV"

    def __init__(self, domain):
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 3:
            raise ValueError("FacetEdgeVectors is only defined for topological dimensions >= 3.")

    @property
    def ufl_shape(self):
        cell = extract_unique_domain(self).ufl_cell()
        facet_types = cell.facet_types()

        # Raise exception for cells with more than one facet type e.g. prisms
        if len(facet_types) > 1:
            raise Exception(f"Cell type {cell} not supported.")

        nfe = facet_types[0].num_edges()
        t = cell.topological_dimension()
        return (nfe, t)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always constant for a given cell type
        return True


@ufl_type()
class CellVertices(GeometricCellQuantity):
    """UFL geometry representation: Physical cell vertices."""
    __slots__ = ()
    name = "CV"

    def __init__(self, domain):
        GeometricCellQuantity.__init__(self, domain)

    @property
    def ufl_shape(self):
        cell = extract_unique_domain(self).ufl_cell()
        nv = cell.num_vertices()
        g = cell.geometric_dimension()
        return (nv, g)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always constant for a given cell type
        return True


@ufl_type()
class CellEdgeVectors(GeometricCellQuantity):
    """UFL geometry representation: The vectors between physical cell vertices for each edge in cell."""
    __slots__ = ()
    name = "CEV"

    def __init__(self, domain):
        GeometricCellQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellEdgeVectors is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        cell = extract_unique_domain(self).ufl_cell()
        ne = cell.num_edges()
        g = cell.geometric_dimension()
        return (ne, g)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always constant for a given cell type
        return True


@ufl_type()
class FacetEdgeVectors(GeometricFacetQuantity):
    """UFL geometry representation: The vectors between physical cell vertices for each edge in current facet."""
    __slots__ = ()
    name = "FEV"

    def __init__(self, domain):
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 3:
            raise ValueError("FacetEdgeVectors is only defined for topological dimensions >= 3.")

    @property
    def ufl_shape(self):
        cell = extract_unique_domain(self).ufl_cell()
        facet_types = cell.facet_types()

        # Raise exception for cells with more than one facet type e.g. prisms
        if len(facet_types) > 1:
            raise Exception(f"Cell type {cell} not supported.")

        nfe = facet_types[0].num_edges()
        g = cell.geometric_dimension()
        return (nfe, g)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always constant for a given cell type
        return True


# --- Determinants (signed or pseudo) of geometry mapping Jacobians

@ufl_type()
class JacobianDeterminant(GeometricCellQuantity):
    """UFL geometry representation: The determinant of the Jacobian.

    Represents the signed determinant of a square Jacobian or the pseudo-determinant of a non-square Jacobian.
    """
    __slots__ = ()
    name = "detJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


@ufl_type()
class FacetJacobianDeterminant(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-determinant of the FacetJacobian."""
    __slots__ = ()
    name = "detFJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


@ufl_type()
class CellFacetJacobianDeterminant(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-determinant of the CellFacetJacobian."""
    __slots__ = ()
    name = "detCFJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


# --- Inverses (signed or pseudo) of geometry mapping Jacobians

@ufl_type()
class JacobianInverse(GeometricCellQuantity):
    """UFL geometry representation: The inverse of the Jacobian.

    Represents the inverse of a square Jacobian or the pseudo-inverse of a non-square Jacobian.
    """
    __slots__ = ()
    name = "K"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension
        of the domain)."""
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (t, g)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


@ufl_type()
class FacetJacobianInverse(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-inverse of the FacetJacobian."""
    __slots__ = ()
    name = "FK"

    def __init__(self, domain):
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("FacetJacobianInverse is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (t - 1, g)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


@ufl_type()
class CellFacetJacobianInverse(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-inverse of the CellFacetJacobian."""
    __slots__ = ()
    name = "CFK"

    def __init__(self, domain):
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellFacetJacobianInverse is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        t = self._domain.topological_dimension()
        return (t - 1, t)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


# --- Types representing normal or tangent vectors

@ufl_type()
class FacetNormal(GeometricFacetQuantity):
    """UFL geometry representation: The outwards pointing normal vector of the current facet."""
    __slots__ = ()
    name = "n"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension
        of the domain)."""
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # For product cells, this is only true for some but not all
        # facets. Seems like too much work to fix right now.  Only
        # true for a piecewise linear coordinate field with simplex
        # _facets_.
        is_piecewise_linear = self._domain.ufl_coordinate_element().degree() == 1
        return is_piecewise_linear and self._domain.ufl_cell().has_simplex_facets()


@ufl_type()
class CellNormal(GeometricCellQuantity):
    """UFL geometry representation: The upwards pointing normal vector of the current manifold cell."""
    __slots__ = ()
    name = "cell_normal"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension
        of the domain)."""
        g = self._domain.geometric_dimension()
        # t = self._domain.topological_dimension()
        # return (g-t,g) # TODO: Should it be CellNormals? For interval in 3D we have two!
        return (g,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


@ufl_type()
class ReferenceNormal(GeometricFacetQuantity):
    """UFL geometry representation: The outwards pointing normal vector of the current facet on the reference cell"""
    __slots__ = ()
    name = "reference_normal"

    @property
    def ufl_shape(self):
        t = self._domain.topological_dimension()
        return (t,)

# TODO: Implement in apply_geometry_lowering and enable
# @ufl_type()
# class FacetTangents(GeometricFacetQuantity):
#     """UFL geometry representation: The tangent vectors of the current facet."""
#     __slots__ = ()
#     name = "t"
#
#     def __init__(self, domain):
#         GeometricFacetQuantity.__init__(self, domain)
#         t = self._domain.topological_dimension()
#         if t < 2:
#             raise ValueError("FacetTangents is only defined for topological dimensions >= 2.")
#
#    @property
#    def ufl_shape(self):
#        g = self._domain.geometric_dimension()
#        t = self._domain.topological_dimension()
#        return (t-1,g)
#
#    def is_cellwise_constant(self): # NB! Copied from FacetNormal
#        "Return whether this expression is spatially constant over each cell."
#        # For product cells, this is only true for some but not all facets. Seems like too much work to fix right now.
#        # Only true for a piecewise linear coordinate field with simplex _facets_.
#        is_piecewise_linear = self._domain.ufl_coordinate_element().degree() == 1
#        return is_piecewise_linear and self._domain.ufl_cell().has_simplex_facets()

# TODO: Implement in apply_geometry_lowering and enable
# @ufl_type()
# class CellTangents(GeometricCellQuantity):
#    """UFL geometry representation: The tangent vectors of the current manifold cell."""
#    __slots__ = ()
#    name = "cell_tangents"
#
#    @property
#    def ufl_shape(self):
#        g = self._domain.geometric_dimension()
#        t = self._domain.topological_dimension()
#        return (t,g)


# --- Types representing midpoint coordinates

# TODO: Implement in the rest of fenics
# @ufl_type()
# class CellMidpoint(GeometricCellQuantity):
#    """UFL geometry representation: The midpoint coordinate of the current cell."""
#    __slots__ = ()
#    name = "cell_midpoint"
#
#    @property
#    def ufl_shape(self):
#        g = self._domain.geometric_dimension()
#        return (g,)

# TODO: Implement in the rest of fenics
# @ufl_type()
# class FacetMidpoint(GeometricFacetQuantity):
#    """UFL geometry representation: The midpoint coordinate of the current facet."""
#    __slots__ = ()
#    name = "facet_midpoint"
#
#    @property
#    def ufl_shape(self):
#        g = self._domain.geometric_dimension()
#        return (g,)


# --- Types representing measures of the cell and entities of the cell, typically used for stabilisation terms

# TODO: Clean up this set of types? Document!


@ufl_type()
class ReferenceCellVolume(GeometricCellQuantity):
    """UFL geometry representation: The volume of the reference cell."""
    __slots__ = ()
    name = "reference_cell_volume"


@ufl_type()
class ReferenceFacetVolume(GeometricFacetQuantity):
    """UFL geometry representation: The volume of the reference cell of the current facet."""
    __slots__ = ()
    name = "reference_facet_volume"


@ufl_type()
class CellVolume(GeometricCellQuantity):
    """UFL geometry representation: The volume of the cell."""
    __slots__ = ()
    name = "volume"


@ufl_type()
class Circumradius(GeometricCellQuantity):
    """UFL geometry representation: The circumradius of the cell."""
    __slots__ = ()
    name = "circumradius"


@ufl_type()
class CellDiameter(GeometricCellQuantity):
    """UFL geometry representation: The diameter of the cell, i.e.,
    maximal distance of two points in the cell."""
    __slots__ = ()
    name = "diameter"


# @ufl_type()
# class CellSurfaceArea(GeometricCellQuantity):
#    """UFL geometry representation: The total surface area of the cell."""
#    __slots__ = ()
#    name = "surfacearea"


@ufl_type()
class FacetArea(GeometricFacetQuantity):  # FIXME: Should this be allowed for interval domain?
    """UFL geometry representation: The area of the facet."""
    __slots__ = ()
    name = "facetarea"


@ufl_type()
class MinCellEdgeLength(GeometricCellQuantity):
    """UFL geometry representation: The minimum edge length of the cell."""
    __slots__ = ()
    name = "mincelledgelength"


@ufl_type()
class MaxCellEdgeLength(GeometricCellQuantity):
    """UFL geometry representation: The maximum edge length of the cell."""
    __slots__ = ()
    name = "maxcelledgelength"


@ufl_type()
class MinFacetEdgeLength(GeometricFacetQuantity):
    """UFL geometry representation: The minimum edge length of the facet."""
    __slots__ = ()
    name = "minfacetedgelength"


@ufl_type()
class MaxFacetEdgeLength(GeometricFacetQuantity):
    """UFL geometry representation: The maximum edge length of the facet."""
    __slots__ = ()
    name = "maxfacetedgelength"


# --- Types representing other stuff

@ufl_type()
class CellOrientation(GeometricCellQuantity):
    """UFL geometry representation: The orientation (+1/-1) of the current cell.

    For non-manifold cells (tdim == gdim), this equals the sign
    of the Jacobian determinant, i.e. +1 if the physical cell is
    oriented the same way as the reference cell and -1 otherwise.

    For manifold cells of tdim==gdim-1 this is input data belonging
    to the mesh, used to distinguish between the sides of the manifold.
    """
    __slots__ = ()
    name = "cell_orientation"


@ufl_type()
class FacetOrientation(GeometricFacetQuantity):
    """UFL geometry representation: The orientation (+1/-1) of the current facet relative to the reference cell."""
    __slots__ = ()
    name = "facet_orientation"


# This doesn't quite fit anywhere. Make a special set of symbolic
# terminal types instead?
@ufl_type()
class QuadratureWeight(GeometricQuantity):
    """UFL geometry representation: The current quadrature weight.

    Only used inside a quadrature context.
    """
    __slots__ = ()
    name = "weight"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # The weight usually varies with the quadrature points
        return False
