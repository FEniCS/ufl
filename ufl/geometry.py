"""Types for representing symbolic expressions for geometric quantities."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.core.terminal import Terminal
from ufl.domain import as_domain, extract_unique_domain
from ufl.restriction import default_restriction, require_restriction
from ufl.sobolevspace import H1

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

class GeometricQuantity(Terminal):
    """Geometric quantity."""

    __slots__ = ("_domain",)

    def __init__(self, domain):
        """Initialise."""
        Terminal.__init__(self)
        self._domain = as_domain(domain)

    def ufl_domains(self):
        """Get the UFL domains."""
        return (self._domain,)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # NB! Geometric quantities are piecewise constant by
        # default. Override if needed.
        return True

    # NB! Geometric quantities are scalar by default. Override if
    # needed.
    ufl_shape = ()

    def _ufl_signature_data_(self, renumbering):
        """Signature data of geometric quantities depend on the domain numbering."""
        return (self._ufl_class_.__name__,) + self._domain._ufl_signature_data_(renumbering)

    def __str__(self):
        """Format as a string."""
        return self._ufl_class_.name

    def __repr__(self):
        """Representation."""
        r = "%s(%s)" % (self._ufl_class_.__name__, repr(self._domain))
        return r

    def _ufl_compute_hash_(self):
        """UFL compute hash."""
        return hash((type(self).__name__,) + self._domain._ufl_hash_data_())

    def __eq__(self, other):
        """Check equality."""
        return isinstance(other, self._ufl_class_) and other._domain == self._domain


class GeometricCellQuantity(GeometricQuantity):
    """Geometric cell quantity."""

    __slots__ = ()

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        return require_restriction(self)


class GeometricFacetQuantity(GeometricQuantity):
    """Geometric facet quantity."""

    __slots__ = ()

    def _ufl_hash_data_(self):
        """Hash data."""
        return (self.__classname__, self._domain)

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        return require_restriction(self)


# --- Coordinate represented in different coordinate systems

class SpatialCoordinate(GeometricCellQuantity):
    """The coordinate in a domain.

    In the context of expression integration,
    represents the domain coordinate of each quadrature point.

    In the context of expression evaluation in a point,
    represents the value of that point.
    """

    __slots__ = ()
    name = "x"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension of the domain)."""
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only case this is true is if the domain is a vertex cell.
        t = self._domain.topological_dimension()
        return t == 0

    def evaluate(self, x, mapping, component, index_values):
        """Return the value of the coordinate."""
        if component == ():
            if isinstance(x, (tuple, list)):
                return float(x[0])
            else:
                return float(x)
        else:
            return float(x[component[0]])

    def count(self):
        """Count."""
        # FIXME: Hack to make SpatialCoordinate behave like a coefficient.
        # When calling `derivative`, the count is used to sort over.
        return -1

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


class CellCoordinate(GeometricCellQuantity):
    """The coordinate in a reference cell.

    In the context of expression integration,
    represents the reference cell coordinate of each quadrature point.

    In the context of expression evaluation in a point in a cell,
    represents that point in the reference coordinate system of the cell.
    """

    __slots__ = ()
    name = "X"

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        t = self._domain.topological_dimension()
        return (t,)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only case this is true is if the domain is a vertex cell.
        t = self._domain.topological_dimension()
        return t == 0


class FacetCoordinate(GeometricFacetQuantity):
    """The coordinate in a reference cell of a facet.

    In the context of expression integration over a facet,
    represents the reference facet coordinate of each quadrature point.

    In the context of expression evaluation in a point on a facet,
    represents that point in the reference coordinate system of the facet.
    """

    __slots__ = ()
    name = "Xf"

    def __init__(self, domain):
        """Initialise."""
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("FacetCoordinate is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        t = self._domain.topological_dimension()
        return (t - 1,)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only case this is true is if the domain is an interval cell
        # (with a vertex facet).
        t = self._domain.topological_dimension()
        return t <= 1

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        return self


# --- Origin of coordinate systems in larger coordinate systems

class CellOrigin(GeometricCellQuantity):
    """The spatial coordinate corresponding to origin of a reference cell."""

    __slots__ = ()
    name = "x0"

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        return True


class FacetOrigin(GeometricFacetQuantity):
    """The spatial coordinate corresponding to origin of a reference facet."""

    __slots__ = ()
    name = "x0f"

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        g = self._domain.geometric_dimension()
        return (g,)

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


class CellFacetOrigin(GeometricFacetQuantity):
    """The reference cell coordinate corresponding to origin of a reference facet."""

    __slots__ = ()
    name = "X0f"

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        t = self._domain.topological_dimension()
        return (t,)


# --- Jacobians of mappings between coordinate systems

class Jacobian(GeometricCellQuantity):
    r"""The Jacobian of the mapping from reference cell to spatial coordinates.

    .. math:: J_{ij} = \\frac{dx_i}{dX_j}
    """

    __slots__ = ()
    name = "J"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension of the domain)."""
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (g, t)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


class FacetJacobian(GeometricFacetQuantity):
    """The Jacobian of the mapping from reference facet to spatial coordinates.

      FJ_ij = dx_i/dXf_j

    The FacetJacobian is the product of the Jacobian and CellFacetJacobian:

      FJ = dx/dXf = dx/dX dX/dXf = J * CFJ
    """

    __slots__ = ()
    name = "FJ"

    def __init__(self, domain):
        """Initialise."""
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("FacetJacobian is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (g, t - 1)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


class CellFacetJacobian(GeometricFacetQuantity):  # dX/dXf
    """The Jacobian of the mapping from reference facet to reference cell coordinates.

    CFJ_ij = dX_i/dXf_j
    """

    __slots__ = ()
    name = "CFJ"

    def __init__(self, domain):
        """Initialise."""
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellFacetJacobian is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        t = self._domain.topological_dimension()
        return (t, t - 1)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # This is always a constant mapping between two reference
        # coordinate systems.
        return True


class ReferenceCellEdgeVectors(GeometricCellQuantity):
    """The vectors between reference cell vertices for each edge in cell."""

    __slots__ = ()
    name = "RCEV"

    def __init__(self, domain):
        """Initialise."""
        GeometricCellQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellEdgeVectors is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        cell = extract_unique_domain(self).ufl_cell()
        ne = cell.num_edges()
        t = cell.topological_dimension()
        return (ne, t)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # This is always constant for a given cell type
        return True


class ReferenceFacetEdgeVectors(GeometricFacetQuantity):
    """The vectors between reference cell vertices for each edge in current facet."""

    __slots__ = ()
    name = "RFEV"

    def __init__(self, domain):
        """Initialise."""
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 3:
            raise ValueError("FacetEdgeVectors is only defined for topological dimensions >= 3.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        cell = extract_unique_domain(self).ufl_cell()
        facet_types = cell.facet_types()

        # Raise exception for cells with more than one facet type e.g. prisms
        if len(facet_types) > 1:
            raise Exception(f"Cell type {cell} not supported.")

        nfe = facet_types[0].num_edges()
        t = cell.topological_dimension()
        return (nfe, t)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # This is always constant for a given cell type
        return True


class CellVertices(GeometricCellQuantity):
    """Physical cell vertices."""

    __slots__ = ()
    name = "CV"

    def __init__(self, domain):
        """Initialise."""
        GeometricCellQuantity.__init__(self, domain)

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        cell = extract_unique_domain(self).ufl_cell()
        nv = cell.num_vertices()
        g = cell.geometric_dimension()
        return (nv, g)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # This is always constant for a given cell type
        return True


class CellEdgeVectors(GeometricCellQuantity):
    """The vectors between physical cell vertices for each edge in cell."""

    __slots__ = ()
    name = "CEV"

    def __init__(self, domain):
        """Initialise."""
        GeometricCellQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellEdgeVectors is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        cell = extract_unique_domain(self).ufl_cell()
        ne = cell.num_edges()
        g = cell.geometric_dimension()
        return (ne, g)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # This is always constant for a given cell type
        return True


class FacetEdgeVectors(GeometricFacetQuantity):
    """The vectors between physical cell vertices for each edge in current facet."""

    __slots__ = ()
    name = "FEV"

    def __init__(self, domain):
        """Initialise."""
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 3:
            raise ValueError("FacetEdgeVectors is only defined for topological dimensions >= 3.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        cell = extract_unique_domain(self).ufl_cell()
        facet_types = cell.facet_types()

        # Raise exception for cells with more than one facet type e.g. prisms
        if len(facet_types) > 1:
            raise Exception(f"Cell type {cell} not supported.")

        nfe = facet_types[0].num_edges()
        g = cell.geometric_dimension()
        return (nfe, g)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # This is always constant for a given cell type
        return True


# --- Determinants (signed or pseudo) of geometry mapping Jacobians

class JacobianDeterminant(GeometricCellQuantity):
    """The determinant of the Jacobian.

    Represents the signed determinant of a square Jacobian or the pseudo-determinant of a non-square Jacobian.
    """

    __slots__ = ()
    name = "detJ"

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


class FacetJacobianDeterminant(GeometricFacetQuantity):
    """The pseudo-determinant of the FacetJacobian."""

    __slots__ = ()
    name = "detFJ"

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


class CellFacetJacobianDeterminant(GeometricFacetQuantity):
    """The pseudo-determinant of the CellFacetJacobian."""

    __slots__ = ()
    name = "detCFJ"

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


# --- Inverses (signed or pseudo) of geometry mapping Jacobians

class JacobianInverse(GeometricCellQuantity):
    """The inverse of the Jacobian.

    Represents the inverse of a square Jacobian or the pseudo-inverse of a non-square Jacobian.
    """

    __slots__ = ()
    name = "K"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension of the domain)."""
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (t, g)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()


class FacetJacobianInverse(GeometricFacetQuantity):
    """The pseudo-inverse of the FacetJacobian."""

    __slots__ = ()
    name = "FK"

    def __init__(self, domain):
        """Initialise."""
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("FacetJacobianInverse is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (t - 1, g)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex
        # cells
        return self._domain.is_piecewise_linear_simplex_domain()

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


class CellFacetJacobianInverse(GeometricFacetQuantity):
    """The pseudo-inverse of the CellFacetJacobian."""

    __slots__ = ()
    name = "CFK"

    def __init__(self, domain):
        """Initialise."""
        GeometricFacetQuantity.__init__(self, domain)
        t = self._domain.topological_dimension()
        if t < 2:
            raise ValueError("CellFacetJacobianInverse is only defined for topological dimensions >= 2.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        t = self._domain.topological_dimension()
        return (t - 1, t)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


# --- Types representing normal or tangent vectors

class FacetNormal(GeometricFacetQuantity):
    """The outwards pointing normal vector of the current facet."""

    __slots__ = ()
    name = "n"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension of the domain)."""
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # For product cells, this is only true for some but not all
        # facets. Seems like too much work to fix right now.  Only
        # true for a piecewise linear coordinate field with simplex
        # _facets_.
        ce = self._domain.ufl_coordinate_element()
        is_piecewise_linear = ce.embedded_superdegree <= 1 and ce in H1
        return is_piecewise_linear and self._domain.ufl_cell().has_simplex_facets()

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        domain = self.ufl_domain()
        e = domain.ufl_coordinate_element()
        gdim = domain.geometric_dimension()
        tdim = domain.topological_dimension()
        if e.embedded_superdegree <= 1 and e in H1 and gd == td:
            if side is None:
                require_restriction(self)
            elif side == default_restriction:
                return self(default_restriction)
            else:
                return -self(default_restriction)
        else:
            require_restriction(self)


class CellNormal(GeometricCellQuantity):
    """The upwards pointing normal vector of the current manifold cell."""

    __slots__ = ()
    name = "cell_normal"

    @property
    def ufl_shape(self):
        """Return the number of coordinates defined (i.e. the geometric dimension of the domain)."""
        g = self._domain.geometric_dimension()
        # t = self._domain.topological_dimension()
        # return (g-t,g) # TODO: Should it be CellNormals? For interval in 3D we have two!
        return (g,)

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


class ReferenceNormal(GeometricFacetQuantity):
    """The outwards pointing normal vector of the current facet on the reference cell."""

    __slots__ = ()
    name = "reference_normal"

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        t = self._domain.topological_dimension()
        return (t,)

# --- Types representing measures of the cell and entities of the cell, typically used for stabilisation terms

# TODO: Clean up this set of types? Document!


class ReferenceCellVolume(GeometricCellQuantity):
    """The volume of the reference cell."""

    __slots__ = ()
    name = "reference_cell_volume"

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        return self


class ReferenceFacetVolume(GeometricFacetQuantity):
    """The volume of the reference cell of the current facet."""

    __slots__ = ()
    name = "reference_facet_volume"

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        return self


class CellVolume(GeometricCellQuantity):
    """The volume of the cell."""

    __slots__ = ()
    name = "volume"


class Circumradius(GeometricCellQuantity):
    """The circumradius of the cell."""

    __slots__ = ()
    name = "circumradius"


class CellDiameter(GeometricCellQuantity):
    """The diameter of the cell, i.e., maximal distance of two points in the cell."""

    __slots__ = ()
    name = "diameter"


class FacetArea(GeometricFacetQuantity):  # FIXME: Should this be allowed for interval domain?
    """The area of the facet."""

    __slots__ = ()
    name = "facetarea"

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


class MinCellEdgeLength(GeometricCellQuantity):
    """The minimum edge length of the cell."""

    __slots__ = ()
    name = "mincelledgelength"


class MaxCellEdgeLength(GeometricCellQuantity):
    """The maximum edge length of the cell."""

    __slots__ = ()
    name = "maxcelledgelength"


class MinFacetEdgeLength(GeometricFacetQuantity):
    """The minimum edge length of the facet."""

    __slots__ = ()
    name = "minfacetedgelength"

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


class MaxFacetEdgeLength(GeometricFacetQuantity):
    """The maximum edge length of the facet."""

    __slots__ = ()
    name = "maxfacetedgelength"

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self(default_restriction)


# --- Types representing other stuff

class CellOrientation(GeometricCellQuantity):
    """The orientation (+1/-1) of the current cell.

    For non-manifold cells (tdim == gdim), this equals the sign
    of the Jacobian determinant, i.e. +1 if the physical cell is
    oriented the same way as the reference cell and -1 otherwise.

    For manifold cells of tdim==gdim-1 this is input data belonging
    to the mesh, used to distinguish between the sides of the manifold.
    """

    __slots__ = ()
    name = "cell_orientation"


class FacetOrientation(GeometricFacetQuantity):
    """The orientation (+1/-1) of the current facet relative to the reference cell."""

    __slots__ = ()
    name = "facet_orientation"


# This doesn't quite fit anywhere. Make a special set of symbolic
# terminal types instead?
class QuadratureWeight(GeometricQuantity):
    """The current quadrature weight.

    Only used inside a quadrature context.
    """

    __slots__ = ()
    name = "weight"

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        # The weight usually varies with the quadrature points
        return False

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        return self
