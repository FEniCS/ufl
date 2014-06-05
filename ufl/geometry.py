"Types for representing symbolic expressions for geometric quantities."

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
# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2009
# Modified by Marie E. Rognes 2012

from collections import defaultdict
from ufl.log import warning, error, deprecate
from ufl.assertions import ufl_assert
from ufl.common import istr, EmptyDict
from ufl.terminal import Terminal
from ufl.protocols import id_or_none
from ufl.cell import as_cell, cellname2dim, cell2dim, cellname2facetname, affine_cells, Cell, ProductCell
from ufl.domain import as_domain, Domain, extract_domains, join_domains, ProductDomain

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
    CellCoordinate = JacobianInverse * (SpatialCoordinate - CellOrigio)

Xf = FK * (x - x0f)
    FacetCoordinate = FacetJacobianInverse * (SpatialCoordinate - FacetOrigin)

Xf = CFK * (X - X0f)
    FacetCoordinate = CellFacetJacobianInverse * (CellCoordinate - CellFacetOrigin)

"""


# --- Expression node types

class GeometricQuantity(Terminal):
    __slots__ = ("_domain",)
    def __init__(self, domain):
        Terminal.__init__(self)
        self._domain = as_domain(domain)

    def domains(self):
        return (self._domain,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell (or over each facet for facet quantities)."
        # NB! Geometric quantities are piecewise constant by default. Override if needed.
        return True

    def shape(self):
        "Scalar shaped."
        # NB! Geometric quantities are scalar by default. Override if needed.
        return ()

    def signature_data(self, renumbering):
        "Signature data of geometric quantities depend on the domain numbering."
        return (self._uflclass.__name__,) + self._domain.signature_data(renumbering)

    def __str__(self):
        return self._uflclass.name

    def __repr__(self):
        return "%s(%r)" % (self._uflclass.__name__, self._domain)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((type(self).__name__,) + self._domain.hash_data())
        return self._hash

    def __eq__(self, other):
        return isinstance(other, self._uflclass) and other._domain == self._domain

class GeometricCellQuantity(GeometricQuantity):
    __slots__ = []

class GeometricFacetQuantity(GeometricQuantity):
    __slots__ = []


# --- Coordinate represented in different coordinate systems

class SpatialCoordinate(GeometricCellQuantity):
    """UFL geometry representation: The coordinate in a domain.

    In the context of expression integration,
    represents the domain coordinate of each quadrature point.

    In the context of expression evaluation in a point,
    represents the value of that point.
    """
    __slots__ = ()
    name = "x"

    def shape(self):
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only case this is true is if the domain is a vertex cell.
        t = self._domain.topological_dimension()
        return t == 0

    def evaluate(self, x, mapping, component, index_values):
        if component == ():
            if isinstance(x, (tuple,list)):
                return float(x[0])
            else:
                return float(x)
        else:
            return float(x[component[0]])

class CellCoordinate(GeometricCellQuantity):
    """UFL geometry representation: The coordinate in a reference cell.

    In the context of expression integration,
    represents the reference cell coordinate of each quadrature point.

    In the context of expression evaluation in a point in a cell,
    represents that point in the reference coordinate system of the cell.
    """
    __slots__ = ()
    name = "X"

    def shape(self):
        t = self._domain.topological_dimension()
        return (t,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only case this is true is if the domain is a vertex cell.
        t = self._domain.topological_dimension()
        return t == 0

class FacetCoordinate(GeometricFacetQuantity):
    """UFL geometry representation: The coordinate in a reference cell of a facet.

    In the context of expression integration over a facet,
    represents the reference facet coordinate of each quadrature point.

    In the context of expression evaluation in a point on a facet,
    represents that point in the reference coordinate system of the facet.
    """
    __slots__ = ()
    name = "Xf"

    def shape(self):
        t = self._domain.topological_dimension()
        return (t-1,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only case this is true is if the domain is an interval cell (with a vertex facet).
        t = self._domain.topological_dimension()
        return t <= 1


# --- Origin of coordinate systems in larger coordinate systems

class CellOrigin(GeometricCellQuantity):
    """UFL geometry representation: The spatial coordinate corresponding to origin of a reference cell."""
    __slots__ = ()
    name = "x0"

    def shape(self):
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        return True

class FacetOrigin(GeometricFacetQuantity):
    """UFL geometry representation: The spatial coordinate corresponding to origin of a reference facet."""
    __slots__ = ()
    name = "x0f"

    def shape(self):
        g = self._domain.geometric_dimension()
        return (g,)

class CellFacetOrigin(GeometricFacetQuantity):
    """UFL geometry representation: The reference cell coordinate corresponding to origin of a reference facet."""
    __slots__ = ()
    name = "X0f"

    def shape(self):
        t = self._domain.topological_dimension()
        return (t,)


# --- Jacobians of mappings between coordinate systems

class Jacobian(GeometricCellQuantity):
    """UFL geometry representation: The Jacobian of the mapping from reference cell to spatial coordinates.

    J_ij = dx_i/dX_j
    """
    __slots__ = ()
    name = "J"

    def shape(self):
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (g, t)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class FacetJacobian(GeometricFacetQuantity):
    """UFL geometry representation: The Jacobian of the mapping from reference facet to spatial coordinates.

      FJ_ij = dx_i/dXf_j

    The FacetJacobian is the product of the Jacobian and CellFacetJacobian:

      FJ = dx/dXf = dx/dX dX/dXf = J * CFJ

    """
    __slots__ = ()
    name = "FJ"

    def shape(self):
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (g, t-1)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class CellFacetJacobian(GeometricFacetQuantity): # dX/dXf
    """UFL geometry representation: The Jacobian of the mapping from reference facet to reference cell coordinates.

    CFJ_ij = dX_i/dXf_j
    """
    __slots__ = ()
    name = "CFJ"

    def shape(self):
        t = self._domain.topological_dimension()
        return (t, t-1)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always a constant mapping between two reference coordinate systems.
        return True


# --- Determinants (signed or pseudo) of geometry mapping Jacobians

class JacobianDeterminant(GeometricCellQuantity):
    """UFL geometry representation: The determinant of the Jacobian.

    Represents the signed determinant of a square Jacobian or the pseudo-determinant of a non-square Jacobian.
    """
    __slots__ = ()
    name = "detJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class FacetJacobianDeterminant(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-determinant of the FacetJacobian."""
    __slots__ = ()
    name = "detFJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class CellFacetJacobianDeterminant(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-determinant of the CellFacetJacobian."""
    __slots__ = ()
    name = "detCFJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


# --- Inverses (signed or pseudo) of geometry mapping Jacobians

class JacobianInverse(GeometricCellQuantity):
    """UFL geometry representation: The inverse of the Jacobian.

    Represents the inverse of a square Jacobian or the pseudo-inverse of a non-square Jacobian.
    """
    __slots__ = ()
    name = "K"

    def shape(self):
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (t, g)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class FacetJacobianInverse(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-inverse of the FacetJacobian."""
    __slots__ = ()
    name = "FK"

    def shape(self):
        g = self._domain.geometric_dimension()
        t = self._domain.topological_dimension()
        return (t-1, g)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class CellFacetJacobianInverse(GeometricFacetQuantity):
    """UFL geometry representation: The pseudo-inverse of the CellFacetJacobian."""
    __slots__ = ()
    name = "CFK"

    def shape(self):
        t = self._domain.topological_dimension()
        return (t-1, t)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()


# --- Types representing normal or tangent vectors

class FacetNormal(GeometricFacetQuantity):
    """UFL geometry representation: The outwards pointing normal vector of the current facet."""
    __slots__ = ()
    name = "n"

    def shape(self):
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # TODO: For product cells, this depends on which facet. Seems like too much work to fix right now.
        # Only true for a piecewise linear coordinate field with simplex _facets_
        x = self._domain.coordinates()
        facet_cellname = cellname2facetname.get(self._domain.cell().cellname()) # Allowing None if unknown..
        return (x is None or x.element().degree() == 1) and (facet_cellname in affine_cells) # .. which will become false.

class CellNormal(GeometricCellQuantity):
    """UFL geometry representation: The upwards pointing normal vector of the current manifold cell."""
    __slots__ = ()
    name = "cell_normal"

    def shape(self):
        g = self._domain.geometric_dimension()
        return (g,)

# TODO: Implement in the rest of fenics
#class FacetTangents(GeometricFacetQuantity):
#    """UFL geometry representation: The tangent vectors of the current facet."""
#    __slots__ = ()
#    name = "t"
#
#    def shape(self):
#        g = self._domain.geometric_dimension()
#        t = self._domain.topological_dimension()
#        return (t-1,g)
#
#    def is_cellwise_constant(self): # NB! Copied from FacetNormal
#        "Return whether this expression is spatially constant over each cell."
#        # TODO: For product cells, this depends on which facet. Seems like too much work to fix right now.
#        # Only true for a piecewise linear coordinate field with simplex _facets_
#        x = self._domain.coordinates()
#        facet_cellname = cellname2facetname.get(self._domain.cell().cellname()) # Allowing None if unknown..
#        return (x is None or x.element().degree() == 1) and (facet_cellname in affine_cells) # .. which will become false.

# TODO: Implement in the rest of fenics
#class CellTangents(GeometricCellQuantity):
#    """UFL geometry representation: The tangent vectors of the current manifold cell."""
#    __slots__ = ()
#    name = "cell_tangents"
#
#    def shape(self):
#        g = self._domain.geometric_dimension()
#        t = self._domain.topological_dimension()
#        return (t,g)


# --- Types representing midpoint coordinates

# TODO: Implement in the rest of fenics
#class CellMidpoint(GeometricCellQuantity):
#    """UFL geometry representation: The midpoint coordinate of the current cell."""
#    __slots__ = ()
#    name = "cell_midpoint"
#
#    def shape(self):
#        g = self._domain.geometric_dimension()
#        return (g,)

# TODO: Implement in the rest of fenics
#class FacetMidpoint(GeometricFacetQuantity):
#    """UFL geometry representation: The midpoint coordinate of the current facet."""
#    __slots__ = ()
#    name = "facet_midpoint"
#
#    def shape(self):
#        g = self._domain.geometric_dimension()
#        return (g,)


# --- Types representing measures of the cell and entities of the cell, typically used for stabilisation terms

# TODO: Clean up this set of types? Document!

class CellVolume(GeometricCellQuantity):
    """UFL geometry representation: The volume of the cell."""
    __slots__ = ()
    name = "volume"

class Circumradius(GeometricCellQuantity):
    """UFL geometry representation: The circumradius of the cell."""
    __slots__ = ()
    name = "circumradius"

#class CellSurfaceArea(GeometricCellQuantity):
#    """UFL geometry representation: The total surface area of the cell."""
#    __slots__ = ()
#    name = "surfacearea"

class FacetArea(GeometricFacetQuantity):
    """UFL geometry representation: The area of the facet."""
    __slots__ = ()
    name = "facetarea"

#class FacetDiameter(GeometricFacetQuantity):
#    """UFL geometry representation: The diameter of the facet."""
#    __slots__ = ()
#    name = "facetdiameter"

class MinFacetEdgeLength(GeometricFacetQuantity):
    """UFL geometry representation: The minimum edge length of the facet."""
    __slots__ = ()
    name = "minfacetedgelength"

class MaxFacetEdgeLength(GeometricFacetQuantity):
    """UFL geometry representation: The maximum edge length of the facet."""
    __slots__ = ()
    name = "maxfacetedgelength"


# --- Types representing other stuff

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

class FacetOrientation(GeometricFacetQuantity):
    """UFL geometry representation: The orientation (+1/-1) of the current facet relative to the reference cell."""
    __slots__ = ()
    name = "facet_orientation"

# This doesn't quite fit anywhere. Make a special set of symbolic terminal types instead?
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


# --- Attach deprecated cell properties

# TODO: Remove this deprecated part after a release or two.

def _deprecated_dim(self):
    """The dimension of the cell.

    Only valid if the geometric and topological dimensions are the same.
    """
    deprecate("cell.d is deprecated, please use one of cell.topological_dimension(), cell.geometric_dimension(), domain.topological_dimension() or domain.geometric_dimension() instead.")
    ufl_assert(self.topological_dimension() == self.geometric_dimension(),
               "Cell.d is undefined when geometric and topological dimensions are not the same.")
    return self.geometric_dimension()

def _deprecated_geometric_quantity(name, cls):
    def f(self):
        "UFL geometry value. Deprecated, please use the constructor types instead."
        deprecate("cell.%s is deprecated, please use %s(domain) instead" % (name, cls.__name__))
        return cls(as_domain(self))
    return f

Cell.d = property(_deprecated_dim)
Cell.x = property(_deprecated_geometric_quantity("x", SpatialCoordinate))
Cell.n = property(_deprecated_geometric_quantity("n", FacetNormal))
Cell.volume = property(_deprecated_geometric_quantity("volume", CellVolume))
Cell.circumradius = property(_deprecated_geometric_quantity("circumradius", Circumradius))
Cell.facet_area = property(_deprecated_geometric_quantity("facet_area", FacetArea))
