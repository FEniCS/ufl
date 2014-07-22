"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

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

from six.moves import xrange as range

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.classes import (Terminal, ReferenceGrad, Grad,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         FacetJacobian, FacetJacobianInverse, FacetJacobianDeterminant,
                         CellFacetJacobian,
                         FacetNormal, CellNormal,
                         CellOrientation, FacetOrientation, QuadratureWeight)
from ufl.constantvalue import as_ufl
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.algorithms.analysis import extract_type
from ufl.indexing import Index, indices
from ufl.tensors import as_tensor, as_vector
from ufl.compound_expressions import determinant_expr, cross_expr, inverse_expr
from ufl.operators import sqrt

from ufl.cell import reference_cell_volume


"""
# Some notes:
# Below, let v_i mean physical coordinate of vertex i and V_i mean the reference cell coordinate of the same vertex.


# Add a type for CellVertices? Note that vertices must be computed in linear cell cases!
triangle_vertices[i,j] = component j of vertex i, following ufc numbering conventions


# Add a type for CellEdgeLengths? Note that these are only easy to define in the linear cell case!
# TODO: Check ufc numbering conventions
triangle_edge_lengths    = [v1v2, v0v2, v0v1] # shape (3,)
tetrahedron_edge_lengths = [v0v1, v0v2, v0v3, v1v2, v1v3, v2v3] # shape (6,)


# Here's how to compute edge lengths from the Jacobian:
J =[ [dx0/dX0, dx0/dX1],
     [dx1/dX0, dx1/dX1] ]
# First compute the edge vector, which is constant for each edge: the vector from one vertex to the other
reference_edge_vector_0 = V2 - V1 # Example! Add a type ReferenceEdgeVectors?
# Then apply J to it and take the length of the resulting vector, this is generic for affine cells
edge_length_i = || dot(J, reference_edge_vector_i) ||

e2 = || J[:,0] . < 1, 0> || = || J[:,0] || = || dx/dX0 || = edge length of edge 2 (v0-v1)
e1 = || J[:,1] . < 0, 1> || = || J[:,1] || = || dx/dX1 || = edge length of edge 1 (v0-v2)
e0 = || J[:,:] . <-1, 1> || = || < J[0,1]-J[0,0], J[1,1]-J[1,0] > || = || dx/dX <-1,1> || = edge length of edge 0 (v1-v2)

trev = triangle_reference_edge_vector
evec0 = J00 * trev[edge][0] + J01 * trev[edge][1]
evec1 = J10 * trev[edge][0] + J11 * trev[edge][1]
elen[edge] = sqrt(evec0*evec0 + evec1*evec1)

trev = triangle_reference_edge_vector
evec0 = J00 * trev[edge][0] + J01 * trev[edge][1]
evec1 = J10 * trev[edge][0] + J11 * trev[edge][1]
evec2 = J20 * trev[edge][0] + J21 * trev[edge][1] # Manifold: triangle in 3D
elen[edge] = sqrt(evec0*evec0 + evec1*evec1 + evec2*evec2)

trev = tetrahedron_reference_edge_vector
evec0 = J00 * trev[edge][0] + J01 * trev[edge][1] + J02 * trev[edge][2]
evec1 = J10 * trev[edge][0] + J11 * trev[edge][1] + J12 * trev[edge][2]
evec2 = J20 * trev[edge][0] + J21 * trev[edge][1] + J22 * trev[edge][2]
elen[edge] = sqrt(evec0*evec0 + evec1*evec1 + evec2*evec2)


# Here's how to compute min/max facet edge length:
triangle: == facetarea
tetrahedron: min(elen[edge] for edge in range(6))
or min( min(elen[0], min(elen[1], elen[2])), min(elen[3], min(elen[4], elen[5])) )
(want proper Min/Max types for this!)


# Here's how to compute circumradius for an interval:
circumradius_interval = cellvolume / 2


# Here's how to compute circumradius for a triangle:
e0 = elen[0]
e1 = elen[1]
e2 = elen[2]
circumradius_triangle = (e0*e1*e2) / (4*cellvolume)


# Here's how to compute circumradius for a tetrahedron:
# v1v2 = edge length between vertex 1 and 2
# la,lb,lc = lengths of the sides of an intermediate triangle
la = v1v2 * v0v3
lb = v0v2 * v1v3
lc = v0v1 * v2v3
# p = perimeter
p = (la + lb + lc)
# s = semiperimeter
s = p / 2
# area of intermediate triangle with Herons formula
tmp_area = sqrt(s * (s - la) * (s - lb) * (s - lc))
circumradius_tetrahedron = tmp_area / (6*cellvolume)

"""


class ChangeToReferenceValue(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)

    def form_argument(self, o):
        # Represent 0-derivatives of form arguments on reference element

        element = o.element()

        local_value = PullbackOf(o) # FIXME implement PullbackOf type

        if isinstance(element, FiniteElement):
            S = element.sobolev_space()
            if S == HDiv:
                # Handle HDiv elements with contravariant piola mapping
                # contravariant_hdiv_mapping = (1/det J) * J * PullbackOf(o)
                J = FIXME
                detJ = FIXME
                mapping = (1/detJ) * J
                i, j = indices(2)
                global_value = as_vector(mapping[i, j] * local_value[j], i)
            elif S == HCurl:
                # Handle HCurl elements with covariant piola mapping
                # covariant_hcurl_mapping = JinvT * PullbackOf(o)
                JinvT = FIXME
                mapping = JinvT
                i, j = indices(2)
                global_value = as_vector(mapping[i, j] * local_value[j], i)
            else:
                # Handle the rest with no mapping.
                global_value = local_value
        else:
            error("FIXME: handle mixed element, components need different mappings")

        return global_value

class ChangeToReferenceGrad(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)

    def grad(self, o):
        # FIXME: Handle HDiv elements with contravariant piola mapping specially?
        # FIXME: Handle HCurl elements with covariant piola mapping specially?

        # Peel off the Grads and count them, and get restriction if it's between the grad and the terminal
        ngrads = 0
        restricted = ''
        while not isinstance(o, Terminal):
            if isinstance(o, Grad):
                o, = o.operands()
                ngrads += 1
            elif isinstance(o, Restricted):
                o, = o.operands()
                restricted = o.side()
        f = o

        # Get domain and create Jacobian inverse object
        domain = f.domain()
        Jinv = JacobianInverse(domain)

        # This is an assumption in the below code TODO: Handle grad(grad(.)) for non-affine domains.
        ufl_assert(ngrads == 1 or Jinv.is_cellwise_constant(),
                   "Multiple grads for non-affine domains not currently supported in this algorithm.")

        # Create some new indices
        ii = indices(f.rank()) # Indices to get to the scalar component of f
        jj = indices(ngrads)   # Indices to sum over the local gradient axes with the inverse Jacobian
        kk = indices(ngrads)   # Indices for the leftover inverse Jacobian axes

        # Preserve restricted property
        if restricted:
            Jinv = Jinv(restricted)
            f = f(restricted)

        # Apply the same number of ReferenceGrad without mappings
        lgrad = f
        for i in range(ngrads):
            lgrad = ReferenceGrad(lgrad)

        # Apply mappings with scalar indexing operations (assumes ReferenceGrad(Jinv) is zero)
        jinv_lgrad_f = lgrad[ii+jj]
        for j, k in zip(jj, kk):
            jinv_lgrad_f = Jinv[j, k]*jinv_lgrad_f

        # Wrap back in tensor shape, derivative axes at the end
        jinv_lgrad_f = as_tensor(jinv_lgrad_f, ii+kk)

        return jinv_lgrad_f

    def reference_grad(self, o):
        error("Not expecting reference grad while applying change to reference grad.")

    def coefficient_derivative(self, o):
        error("Coefficient derivatives should be expanded before applying change to reference grad.")

class ChangeToReferenceGeometry(ReuseTransformer):
    def __init__(self, physical_coordinates_known, coordinate_coefficient_mapping):
        ReuseTransformer.__init__(self)
        self.coordinate_coefficient_mapping = coordinate_coefficient_mapping or {} # FIXME: Use this!
        self.physical_coordinates_known = physical_coordinates_known
        self._rcache = {}

    def jacobian(self, o):
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            x = domain.coordinates()
            if x is None:
                r = o
            else:
                x = self.coordinate_coefficient_mapping[x]
                r = ReferenceGrad(x)
            self._rcache[o] = r
        return r

    def jacobian_inverse(self, o):
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            J = self.jacobian(Jacobian(domain))
            r = inverse_expr(J)
            self._rcache[o] = r
        return r

    def jacobian_determinant(self, o):
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            J = self.jacobian(Jacobian(domain))
            r = determinant_expr(J)
            self._rcache[o] = r
        return r

    def facet_jacobian(self, o):
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            J = self.jacobian(Jacobian(domain))
            RFJ = CellFacetJacobian(domain)
            i, j, k = indices(3)
            r = as_tensor(J[i, k]*RFJ[k, j], (i, j))
            self._rcache[o] = r
        return r

    def facet_jacobian_inverse(self, o):
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            FJ = self.facet_jacobian(FacetJacobian(domain))
            r = inverse_expr(FJ)
            self._rcache[o] = r
        return r

    def facet_jacobian_determinant(self, o):
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            FJ = self.facet_jacobian(FacetJacobian(domain))
            r = determinant_expr(FJ)
            self._rcache[o] = r
        return r

    def spatial_coordinate(self, o):
        "Fall through to coordinate field of domain if it exists."
        if self.physical_coordinates_known:
            return o
        else:
            domain = o.domain()
            x = domain.coordinates()
            if x is None:
                return o
            else:
                x = self.coordinate_coefficient_mapping[x]
                return x

    def cell_coordinate(self, o):
        "Compute from physical coordinates if they are known, using the appropriate mappings."
        if self.physical_coordinates_known:
            r = self._rcache.get(o)
            if r is None:
                K = self.jacobian_inverse(JacobianInverse(domain))
                x = self.spatial_coordinate(SpatialCoordinate(domain))
                x0 = CellOrigin(domain)
                i, j = indices(2)
                X = as_tensor(K[i, j] * (x[j] - x0[j]), (i,))
                r = X
            return r
        else:
            return o

    def facet_cell_coordinate(self, o):
        if self.physical_coordinates_known:
            error("Missing computation of facet reference coordinates from physical coordinates via mappings.")
        else:
            return o

    def cell_volume(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the cell volume of an affine cell.")
        r = self.jacobian_determinant(JacobianDeterminant(domain))
        r0 = reference_cell_volume[domain.cell().cellname()]
        return abs(r * r0)

    def facet_area(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the facet area of an affine cell.")
        r = self.facet_jacobian_determinant(FacetJacobianDeterminant(domain))
        r0 = reference_cell_volume[domain.cell().facet_cellname()]
        return abs(r * r0)

    def cell_normal(self, o):
        warning("Untested complicated code for cell normal. Please report if this works correctly or not.")

        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            gdim = domain.geometric_dimension()
            tdim = domain.topological_dimension()

            if tdim == gdim - 1:

                if tdim == 2: # Surface in 3D
                    J = self.jacobian(Jacobian(domain))
                    cell_normal = cross_expr(J[:, 0], J[:, 1])

                elif tdim == 1: # Line in 2D
                    # TODO: Sign here is ambiguous, which normal?
                    cell_normal = as_vector((J[1, 0], -J[0, 0]))

                i = Index()
                cell_normal = cell_normal / sqrt(cell_normal[i]*cell_normal[i])

            elif tdim == gdim:
                cell_normal = as_vector((0.0,)*tdim + (1.0,))

            else:
                error("What do you mean by cell normal in gdim={0}, tdim={1}?".format(gdim, tdim))

            r = cell_normal
        return r

    def facet_normal(self, o):
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            gdim = domain.geometric_dimension()
            tdim = domain.topological_dimension()

            if tdim == 3:
                FJ = self.facet_jacobian(FacetJacobian(domain))

                ufl_assert(gdim == 3, "Inconsistent dimensions.")
                ufl_assert(FJ.shape() == (3, 2), "Inconsistent dimensions.")

                # Compute signed scaling factor
                scale = self.jacobian_determinant(JacobianDeterminant(domain))

                # Compute facet normal direction of 3D cell, product of two tangent vectors
                fo = FacetOrientation(domain)
                ndir = (fo * scale) * cross_expr(FJ[:, 0], FJ[:, 1])

                # Normalise normal vector
                i = Index()
                n = ndir / sqrt(ndir[i]*ndir[i])
                r = n

            elif tdim == 2:
                FJ = self.facet_jacobian(FacetJacobian(domain))

                if gdim == 2:
                    # 2D facet normal in 2D space
                    ufl_assert(FJ.shape() == (2, 1), "Inconsistent dimensions.")

                    # Compute facet tangent
                    tangent = as_vector((FJ[0, 0], FJ[1, 0], 0))

                    # Compute cell normal
                    cell_normal = as_vector((0, 0, 1))

                    # Compute signed scaling factor
                    scale = self.jacobian_determinant(JacobianDeterminant(domain))
                else:
                    # 2D facet normal in 3D space
                    ufl_assert(FJ.shape() == (gdim, 1), "Inconsistent dimensions.")

                    # Compute facet tangent
                    tangent = FJ[:, 0]

                    # Compute cell normal
                    cell_normal = self.cell_normal(CellNormal(domain))

                    # Compute signed scaling factor (input in the manifold case)
                    scale = CellOrientation(domain)

                ufl_assert(len(tangent) == 3, "Inconsistent dimensions.")
                ufl_assert(len(cell_normal) == 3, "Inconsistent dimensions.")

                # Compute normal direction
                cr = cross_expr(tangent, cell_normal)
                if gdim == 2:
                    cr = as_vector((cr[0], cr[1]))
                fo = FacetOrientation(domain)
                ndir = (fo * scale) * cr

                # Normalise normal vector
                i = Index()
                n = ndir / sqrt(ndir[i]*ndir[i])
                r = n

            elif tdim == 1:
                J = self.jacobian(Jacobian(domain)) # dx/dX
                fo = FacetOrientation(domain)
                ndir = fo * J[:,0]
                if gdim == 1:
                    nlen = abs(ndir[0])
                else:
                    i = Index()
                    nlen = sqrt(ndir[i]*ndir[i])
                n = ndir / nlen
                r = n

            self._rcache[o] = r

        ufl_assert(r.shape() == o.shape(), "Inconsistent dimensions (in=%d, out=%d)." % (o.shape()[0], r.shape()[0]))
        return r


def change_to_reference_grad(e):
    """Change Grad objects in expression to products of JacobianInverse and ReferenceGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    return apply_transformer(e, ChangeToReferenceGrad())


def change_to_reference_geometry(e, physical_coordinates_known, coordinate_coefficient_mapping=None):
    """Change GeometricQuantity objects in expression to the lowest level GeometricQuantity objects.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    return apply_transformer(e, ChangeToReferenceGeometry(physical_coordinates_known, coordinate_coefficient_mapping))


def compute_integrand_scaling_factor(domain, integral_type):
    """Change integrand geometry to the right representations."""

    weight = QuadratureWeight(domain)
    tdim = domain.topological_dimension()

    if integral_type == "cell":
        scale = abs(JacobianDeterminant(domain)) * weight

    elif integral_type in ["exterior_facet", "exterior_facet_bottom", "exterior_facet_top", "exterior_facet_vert"]:
        if tdim > 1:
            # Scaling integral by facet jacobian determinant and quadrature weight
            scale = FacetJacobianDeterminant(domain) * weight
        else:
            # No need to scale 'integral' over a vertex
            scale = 1

    elif integral_type in ["interior_facet", "interior_facet_horiz", "interior_facet_vert"]:
        if tdim > 1:
            # Scaling integral by facet jacobian determinant from one side and quadrature weight
            scale = FacetJacobianDeterminant(domain)('+') * weight
        else:
            # No need to scale 'integral' over a vertex
            scale = 1

    elif integral_type == "custom":
        # Scaling with custom weight, which includes eventual volume scaling
        scale = weight

    elif integral_type == "point":
        # No need to scale 'integral' over a point
        scale = 1

    return scale


def change_integrand_geometry_representation(integrand, scale, integral_type):
    """Change integrand geometry to the right representations."""

    # FIXME: We have a serious problem here:
    # - Applying propagate_restrictions before change_integrand_geometry_representation will fail because we need grad(u)('+') not grad(u('+'))
    # - Applying propagate_restrictions after change_integrand_geometry_representation will fail to set e.g. n('+') = -n('-') because n is rewritten
    # - Possible solution: treat grad(u('+')) in grad handler above as grad(u)('+') and apply propagate_restrictions first

    integrand = change_to_reference_grad(integrand)

    integrand = integrand * scale

    if integral_type == "quadrature":
        physical_coordinates_known = True
    else:
        physical_coordinates_known = False
    integrand = change_to_reference_geometry(integrand, physical_coordinates_known)

    return integrand
