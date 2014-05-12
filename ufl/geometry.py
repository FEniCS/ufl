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
from ufl.cell import as_cell, cellname2dim, cellname2facetname, affine_cells, Cell, ProductCell
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
    PhysicalCellJacobian

Jxf = dx/dXf = grad_Xf x(Xf)  =  Jxc Jcf = dx/dX dX/dXf = grad_X x(X) grad_Xf X(Xf)
    PhysicalFacetJacobian = PhysicalCellJacobian * CellFacetJacobian


Possible computation of X from Xf:

X = Jcf Xf + X0f
    CellCoordinate = CellFacetJacobian * FacetCoordinate + CellFacetOrigin


Possible computation of x from X:

x = f(X)
    SpatialCoordinate = sum_k xdofs_k * xphi_k(X)

x = Jxc X + x0
    SpatialCoordinate = PhysicalCellJacobian * CellCoordinate + PhysicalCellOrigo


Possible computation of x from Xf:

x = Jxf Xf + x0f
    SpatialCoordinate = PhysicalFacetJacobian * FacetCoordinate + PhysicalFacetOrigin

x = Jxc Jcf Xf + x0f
    SpatialCoordinate = PhysicalCellJacobian * CellFacetJacobian * FacetCoordinate + PhysicalFacetOrigin


Names:

s/PhysicalCellOrigo/PhysicalCellOrigo/g

s/ReferenceFacetJacobian/CellFacetJacobian/g
s/FacetJacobian/PhysicalFacetJacobian/g

...careful... 's/\<Jacobian\>/PhysicalCellJacobian/g'

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
        "Return whether this expression is spatially constant over each cell."
        # NB! Geometric quantities are piecewise constant by default. Override if needed.
        return True

    def shape(self):
        "Scalar shaped."
        # NB! Geometric quantities are scalar by default. Override if needed.
        return ()

    def signature_data(self, domain_numbering):
        "Signature data of geometric quantities depend on the domain numbering."
        return (type(self).__name__,) + self._domain.signature_data(domain_numbering)

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

class SpatialCoordinate(GeometricCellQuantity): # x
    "Representation of a spatial coordinate."
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

class CellCoordinate(GeometricCellQuantity): # X
    "Representation of a local coordinate on the reference cell."
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

class FacetCoordinate(GeometricFacetQuantity): # Xf
    "Representation of a local coordinate on the reference cell of (a facet of the reference cell)."
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

class PhysicalCellOrigo(GeometricCellQuantity): # x0
    "Representation of the physical coordinate corresponding to the origin on the reference cell."
    __slots__ = ()
    name = "x0"

    def shape(self):
        g = self._domain.geometric_dimension()
        return (g,)

    def is_cellwise_constant(self):
        return True

# TODO: Add to init and classes and algorithms
#class CellFacetOrigin(GeometricFacetQuantity): # X0f
#    "Representation of the reference cell coordinate corresponding to the origin of the reference facet."
#    __slots__ = ()
#    name = "X0f"
#
#    def shape(self):
#        t = self._domain.topological_dimension()
#        return (t,)

#class PhysicalFacetOrigin(GeometricFacetQuantity): # x0f
#    "Representation of the physical coordinate corresponding to the origin of the reference facet."
#    __slots__ = ()
#    name = "X0f"
#
#    def shape(self):
#        g = self._domain.geometric_dimension()
#        return (g,)


class Jacobian(GeometricCellQuantity): # dx/dX
    "Representation of the Jacobian of the mapping from reference cell to physical coordinates."
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

class JacobianDeterminant(GeometricCellQuantity): # det(dx/dX)
    "Representation of the (pseudo-)determinant of the Jacobian of the mapping from reference cell to physical coordinates."
    __slots__ = ()
    name = "detJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class JacobianInverse(GeometricCellQuantity): # inv(dx/dX)
    "Representation of the (pseudo-)inverse of the Jacobian of the mapping from reference cell to physical coordinates."
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


class FacetJacobian(GeometricFacetQuantity): # dx/dXf = dx/dX dX/dXf
    "Representation of the Jacobian of the mapping from reference cell of facet to physical coordinates."
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

class FacetJacobianDeterminant(GeometricFacetQuantity): # det(dx/dXf)
    "Representation of the (pseudo-)determinant of the Jacobian of the mapping from reference cell of facet to physical coordinates."
    __slots__ = ()
    name = "detFJ"

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # Only true for a piecewise linear coordinate field in simplex cells
        return self._domain.is_piecewise_linear_simplex_domain()

class FacetJacobianInverse(iGeometricFacetQuantity): # inv(dx/dXf)
    "Representation of the (pseudo-)inverse of the Jacobian of the mapping from reference cell of facet to physical coordinates."
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


class ReferenceFacetJacobian(GeometricFacetQuantity): # dX/dXf
    "Representation of the Jacobian of the mapping from (coordinates on reference cell of facet) to (cell reference coordinates on facet of reference cell)."
    __slots__ = ()
    name = "RFJ"

    def shape(self):
        t = self._domain.topological_dimension()
        return (t, t-1)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        # This is always a constant mapping between two reference coordinate systems.
        return True

#class ReferenceFacetJacobianDeterminant(GeometricFacetQuantity): # det(dX/dXf)
#    "Representation of the (pseudo-)determinant of the Jacobian of the mapping from (coordinates on reference cell of facet) to (cell reference coordinates on facet of reference cell)."
#    __slots__ = ()
#    name = "detRFJ"
#
#    def is_cellwise_constant(self):
#        "Return whether this expression is spatially constant over each cell."
#        # Only true for a piecewise linear coordinate field in simplex cells
#        return self._domain.is_piecewise_linear_simplex_domain()

#class ReferenceFacetJacobianInverse(GeometricFacetQuantity): # inv(dX/dXf)
#    "Representation of the (pseudo-)inverse of the Jacobian of the mapping from (coordinates on reference cell of facet) to (cell reference coordinates on facet of reference cell)."
#    __slots__ = ()
#    name = "RFK"
#
#    def shape(self):
#        t = self._domain.topological_dimension()
#        return (t-1, t)
#
#    def is_cellwise_constant(self):
#        "Return whether this expression is spatially constant over each cell."
#        # Only true for a piecewise linear coordinate field in simplex cells
#        return self._domain.is_piecewise_linear_simplex_domain()


# --- Types representing normal vectors

class FacetNormal(GeometricFacetQuantity):
    "Representation of a facet normal."
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
    "Representation of a cell normal, for cells of tdim=gdim-1."
    __slots__ = ()
    name = "cell_normal"

    def shape(self):
        g = self._domain.geometric_dimension()
        return (g,)


# TODO: CellTangents (t,g), FacetTangents (t-1,g)


# --- Types representing barycenter coordinates

#class CellBarycenter(GeometricCellQuantity):
#    "Representation of the physical barycenter coordinate of the cell."
#    __slots__ = ()
#    name = "cell_barycenter"
#
#    def shape(self):
#        g = self._domain.geometric_dimension()
#        return (g,)

#class FacetBarycenter(GeometricFacetQuantity):
#    "Representation of the physical barycenter coordinate of the facet."
#    __slots__ = ()
#    name = "facet_barycenter"
#
#    def shape(self):
#        g = self._domain.geometric_dimension()
#        return (g,)


# --- Types representing measures of the cell and entities of the cell, typically used for stabilisation terms

class CellVolume(GeometricCellQuantity):
    "Representation of a cell volume."
    __slots__ = ()
    name = "volume"

class Circumradius(GeometricCellQuantity):
    "Representation of the circumradius of a cell."
    __slots__ = ()
    name = "circumradius"

#class CellSurfaceArea(GeometricCellQuantity):
#    "Representation of the total surface area of a cell."
#    __slots__ = ()
#    name = "surfacearea"

class FacetArea(GeometricFacetQuantity):
    "Representation of the area of a cell facet."
    __slots__ = ()
    name = "facetarea"

#class FacetDiameter(GeometricFacetQuantity):
#    """(EXPERIMENTAL) Representation of the diameter of a facet.
#
#    This is not yet defined.
#    """
#    __slots__ = ()
#    name = "facetdiameter"

class MinFacetEdgeLength(GeometricFacetQuantity):
    "Representation of the minimum edge length of a facet."
    __slots__ = ()
    name = "minfacetedgelength"

class MaxFacetEdgeLength(GeometricFacetQuantity):
    "Representation of the maximum edge length of a facet."
    __slots__ = ()
    name = "maxfacetedgelength"


# --- Types representing other stuff

class CellOrientation(GeometricCellQuantity):
    """Representation of cell orientation, for cells of tdim==gdim-1.

    For affine cells with tdim==gdim, this equals the sign of the Jacobian determinant.

    For non-affine cells... you probably need to figure that out yourself and send a patch.
    """
    __slots__ = ()
    name = "cell_orientation"

# This doesn't quite fit anywhere. Make a special set of symbolic terminal types instead?
class QuadratureWeight(GeometricQuantity):
    "Representation of the current quadrature weight. Only used inside a quadrature context."
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
