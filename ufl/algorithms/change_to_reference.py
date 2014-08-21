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

from ufl.core.multiindex import Index, indices
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag

from ufl.classes import (Terminal, ReferenceGrad, Grad, Restricted, ReferenceValue,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         FacetJacobian, FacetJacobianInverse, FacetJacobianDeterminant,
                         CellFacetJacobian,
                         CellEdgeVectors, FacetEdgeVectors,
                         FacetNormal, CellNormal,
                         CellVolume, FacetArea,
                         CellOrientation, FacetOrientation, QuadratureWeight)

from ufl.constantvalue import as_ufl
from ufl.tensors import as_tensor, as_vector
from ufl.operators import sqrt, max_value, min_value

from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.compound_expressions import determinant_expr, cross_expr, inverse_expr


# TODO: Move to ufl.corealg.multifunction?
def memoized_handler(handler, cachename="_cache"):
    def _memoized_handler(self, o):
        c = getattr(self, cachename)
        r = c.get(o)
        if r is None:
            r = handler(self, o)
            c[o] = r
        return r
    return _memoized_handler



"""
# Some notes:
# Below, let v_i mean physical coordinate of vertex i and V_i mean the reference cell coordinate of the same vertex.


# Add a type for CellVertices? Note that vertices must be computed in linear cell cases!
triangle_vertices[i,j] = component j of vertex i, following ufc numbering conventions


# DONE Add a type for CellEdgeLengths? Note that these are only easy to define in the linear cell case!
triangle_edge_lengths    = [v1v2, v0v2, v0v1] # shape (3,)
tetrahedron_edge_lengths = [v0v1, v0v2, v0v3, v1v2, v1v3, v2v3] # shape (6,)


# DONE Here's how to compute edge lengths from the Jacobian:
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
evec0 = J00 * trev[edge][0] + J01 * trev[edge][1]  =  J*trev[edge]
evec1 = J10 * trev[edge][0] + J11 * trev[edge][1]
elen[edge] = sqrt(evec0*evec0 + evec1*evec1)  = sqrt((J*trev[edge])**2)

trev = triangle_reference_edge_vector
evec0 = J00 * trev[edge][0] + J01 * trev[edge][1]  =  J*trev
evec1 = J10 * trev[edge][0] + J11 * trev[edge][1]
evec2 = J20 * trev[edge][0] + J21 * trev[edge][1] # Manifold: triangle in 3D
elen[edge] = sqrt(evec0*evec0 + evec1*evec1 + evec2*evec2)  = sqrt((J*trev[edge])**2)

trev = tetrahedron_reference_edge_vector
evec0 = sum(J[0,k] * trev[edge][k] for k in range(3))
evec1 = sum(J[1,k] * trev[edge][k] for k in range(3))
evec2 = sum(J[2,k] * trev[edge][k] for k in range(3))
elen[edge] = sqrt(evec0*evec0 + evec1*evec1 + evec2*evec2)  = sqrt((J*trev[edge])**2)


# DONE Here's how to compute min/max facet edge length:
triangle:
  r = facetarea
tetrahedron:
  min(elen[edge] for edge in range(6))
or
  min( min(elen[0], min(elen[1], elen[2])), min(elen[3], min(elen[4], elen[5])) )
or
  min1 = min_value(elen[0], min_value(elen[1], elen[2]))
  min2 = min_value(elen[3], min_value(elen[4], elen[5]))
  r = min_value(min1, min2)
(want proper Min/Max types for this!)


# DONE Here's how to compute circumradius for an interval:
circumradius_interval = cellvolume / 2


# DONE Here's how to compute circumradius for a triangle:
e0 = elen[0]
e1 = elen[1]
e2 = elen[2]
circumradius_triangle = (e0*e1*e2) / (4*cellvolume)


# DONE Here's how to compute circumradius for a tetrahedron:
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

        local_value = ReferenceValue(o)

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

#class ChangeToReferenceGrad(ReuseTransformer):
#    def __init__(self):
#        ReuseTransformer.__init__(self)
class ChangeToReferenceGrad(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        #self._cache = {}

    def expr(self, o, *ops):
        return o.reconstruct(*ops)

    def terminal(self, o):
        return o

    def grad(self, o, dummy_op):
        # FIXME: Handle HDiv elements with contravariant piola mapping specially?
        # FIXME: Handle HCurl elements with covariant piola mapping specially?

        # Peel off the Grads and count them, and get restriction if it's between the grad and the terminal
        ngrads = 0
        restricted = ''
        while not o._ufl_is_terminal_:
            if isinstance(o, Grad):
                o, = o.ufl_operands
                ngrads += 1
            elif isinstance(o, Restricted):
                restricted = o.side()
                o, = o.ufl_operands
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

    def reference_grad(self, o, dummy_op):
        error("Not expecting reference grad while applying change to reference grad.")

    def coefficient_derivative(self, o, *dummy_ops):
        error("Coefficient derivatives should be expanded before applying change to reference grad.")


class ChangeToReferenceGeometry(MultiFunction):
    def __init__(self, physical_coordinates_known, coordinate_coefficient_mapping):
        MultiFunction.__init__(self)
        self.coordinate_coefficient_mapping = coordinate_coefficient_mapping or {}
        self.physical_coordinates_known = physical_coordinates_known
        self._cache = {}

    def expr(self, o, *ops):
        return o.reconstruct(*ops)

    def terminal(self, o):
        return o

    @memoized_handler
    def jacobian(self, o):
        domain = o.domain()
        x = domain.coordinates()
        if x is None:
            r = o
        else:
            x = self.coordinate_coefficient_mapping[x]
            r = ReferenceGrad(x)
        return r

    @memoized_handler
    def jacobian_inverse(self, o):
        domain = o.domain()
        J = self.jacobian(Jacobian(domain))
        return inverse_expr(J)

    @memoized_handler
    def jacobian_determinant(self, o):
        domain = o.domain()
        J = self.jacobian(Jacobian(domain))
        return determinant_expr(J)

    @memoized_handler
    def facet_jacobian(self, o):
        domain = o.domain()
        J = self.jacobian(Jacobian(domain))
        RFJ = CellFacetJacobian(domain)
        i, j, k = indices(3)
        return as_tensor(J[i, k]*RFJ[k, j], (i, j))

    @memoized_handler
    def facet_jacobian_inverse(self, o):
        domain = o.domain()
        FJ = self.facet_jacobian(FacetJacobian(domain))
        return inverse_expr(FJ)

    @memoized_handler
    def facet_jacobian_determinant(self, o):
        domain = o.domain()
        FJ = self.facet_jacobian(FacetJacobian(domain))
        return determinant_expr(FJ)

    @memoized_handler
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

    @memoized_handler
    def cell_coordinate(self, o):
        "Compute from physical coordinates if they are known, using the appropriate mappings."
        if self.physical_coordinates_known:
            K = self.jacobian_inverse(JacobianInverse(domain))
            x = self.spatial_coordinate(SpatialCoordinate(domain))
            x0 = CellOrigin(domain)
            i, j = indices(2)
            X = as_tensor(K[i, j] * (x[j] - x0[j]), (i,))
            return X
        else:
            return o

    @memoized_handler
    def facet_cell_coordinate(self, o):
        if self.physical_coordinates_known:
            error("Missing computation of facet reference coordinates from physical coordinates via mappings.")
        else:
            return o

    @memoized_handler
    def cell_volume(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the cell volume of an affine cell.")
        r = self.jacobian_determinant(JacobianDeterminant(domain))
        r0 = domain.cell().reference_volume()
        return abs(r * r0)

    @memoized_handler
    def facet_area(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the facet area of an affine cell.")
        r = self.facet_jacobian_determinant(FacetJacobianDeterminant(domain))
        r0 = domain.cell().reference_facet_volume()
        return abs(r * r0)

    @memoized_handler
    def circumradius(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the circumradius of an affine cell.")
        cellname = domain.cell().cellname()
        cellvolume = self.cell_volume(CellVolume(domain))

        if cellname == "interval":
            r = 0.5 * cellvolume

        elif cellname == "triangle":
            J = self.jacobian(Jacobian(domain))
            trev = CellEdgeVectors(domain)
            num_edges = 3
            i, j, k = indices(3)
            elen = [sqrt((J[i, j]*trev[edge, j])*(J[i, k]*trev[edge, k])) for edge in range(num_edges)]

            r = (elen[0] * elen[1] * elen[2]) / (4.0 * cellvolume)

        elif cellname == "tetrahedron":
            J = self.jacobian(Jacobian(domain))
            trev = CellEdgeVectors(domain)
            num_edges = 6
            i, j, k = indices(3)
            elen = [sqrt((J[i, j]*trev[edge, j])*(J[i, k]*trev[edge, k])) for edge in range(num_edges)]

            # elen[3] = length of edge 3
            # la, lb, lc = lengths of the sides of an intermediate triangle
            la = elen[3] * elen[2]
            lb = elen[4] * elen[1]
            lc = elen[5] * elen[0]
            # p = perimeter
            p = (la + lb + lc)
            # s = semiperimeter
            s = p / 2
            # area of intermediate triangle with Herons formula
            triangle_area = sqrt(s * (s - la) * (s - lb) * (s - lc))
            r = triangle_area / (6.0 * cellvolume)

        else:
            error("Unhandled cell type %s." % cellname)

        return r

    @memoized_handler
    def min_cell_edge_length(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the min_cell_edge_length of an affine cell.")
        cellname = domain.cell().cellname()

        J = self.jacobian(Jacobian(domain))
        trev = CellEdgeVectors(domain)
        num_edges = trev.ufl_shape[0]
        i, j, k = indices(3)
        elen = [sqrt((J[i, j]*trev[edge, j])*(J[i, k]*trev[edge, k])) for edge in range(num_edges)]

        if cellname == "triangle":
            return min_value(elen[0], min_value(elen[1], elen[2]))
        elif cellname == "tetrahedron":
            min1 = min_value(elen[0], min_value(elen[1], elen[2]))
            min2 = min_value(elen[3], min_value(elen[4], elen[5]))
            return min_value(min1, min2)
        else:
            error("Unhandled cell type %s." % cellname)

    @memoized_handler
    def max_cell_edge_length(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the max_cell_edge_length of an affine cell.")
        cellname = domain.cell().cellname()

        J = self.jacobian(Jacobian(domain))
        trev = CellEdgeVectors(domain)
        num_edges = trev.ufl_shape[0]
        i, j, k = indices(3)
        elen = [sqrt((J[i, j]*trev[edge, j])*(J[i, k]*trev[edge, k])) for edge in range(num_edges)]

        if cellname == "triangle":
            return max_value(elen[0], max_value(elen[1], elen[2]))
        elif cellname == "tetrahedron":
            max1 = max_value(elen[0], max_value(elen[1], elen[2]))
            max2 = max_value(elen[3], max_value(elen[4], elen[5]))
            return max_value(max1, max2)
        else:
            error("Unhandled cell type %s." % cellname)

    @memoized_handler
    def min_facet_edge_length(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the min_facet_edge_length of an affine cell.")
        cellname = domain.cell().cellname()

        if cellname == "triangle":
            return self.facet_area(FacetArea(domain))
        elif cellname == "tetrahedron":
            J = self.jacobian(Jacobian(domain))
            trev = FacetEdgeVectors(domain)
            num_edges = 3
            i, j, k = indices(3)
            elen = [sqrt((J[i, j]*trev[edge, j])*(J[i, k]*trev[edge, k])) for edge in range(num_edges)]
            return min_value(elen[0], min_value(elen[1], elen[2]))
        else:
            error("Unhandled cell type %s." % cellname)

    @memoized_handler
    def max_facet_edge_length(self, o):
        domain = o.domain()
        if not domain.is_piecewise_linear_simplex_domain():
            error("Only know how to compute the max_facet_edge_length of an affine cell.")
        cellname = domain.cell().cellname()

        if cellname == "triangle":
            return self.facet_area(FacetArea(domain))
        elif cellname == "tetrahedron":
            J = self.jacobian(Jacobian(domain))
            trev = FacetEdgeVectors(domain)
            num_edges = 3
            i, j, k = indices(3)
            elen = [sqrt((J[i, j]*trev[edge, j])*(J[i, k]*trev[edge, k])) for edge in range(num_edges)]
            return max_value(elen[0], max_value(elen[1], elen[2]))
        else:
            error("Unhandled cell type %s." % cellname)

    @memoized_handler
    def cell_normal(self, o):
        warning("Untested complicated code for cell normal. Please report if this works correctly or not.")

        domain = o.domain()
        gdim = domain.geometric_dimension()
        tdim = domain.topological_dimension()

        if tdim == gdim - 1:
            if tdim == 2: # Surface in 3D
                J = self.jacobian(Jacobian(domain))
                cell_normal = cross_expr(J[:, 0], J[:, 1])
            elif tdim == 1: # Line in 2D
                # TODO: Document which normal direction this is
                cell_normal = as_vector((-J[1, 0], J[0, 0]))
            i = Index()
            return cell_normal / sqrt(cell_normal[i]*cell_normal[i])
        elif tdim == gdim:
            return as_vector((0.0,)*tdim + (1.0,))
        else:
            error("What do you mean by cell normal in gdim={0}, tdim={1}?".format(gdim, tdim))

    @memoized_handler
    def facet_normal(self, o):
        domain = o.domain()
        gdim = domain.geometric_dimension()
        tdim = domain.topological_dimension()

        if tdim == 3:
            FJ = self.facet_jacobian(FacetJacobian(domain))

            ufl_assert(gdim == 3, "Inconsistent dimensions.")
            ufl_assert(FJ.ufl_shape == (3, 2), "Inconsistent dimensions.")

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
                ufl_assert(FJ.ufl_shape == (2, 1), "Inconsistent dimensions.")

                # Compute facet tangent
                tangent = as_vector((FJ[0, 0], FJ[1, 0], 0))

                # Compute cell normal
                cell_normal = as_vector((0, 0, 1))

                # Compute signed scaling factor
                scale = self.jacobian_determinant(JacobianDeterminant(domain))
            else:
                # 2D facet normal in 3D space
                ufl_assert(FJ.ufl_shape == (gdim, 1), "Inconsistent dimensions.")

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
            ndir = fo * J[:, 0]
            if gdim == 1:
                nlen = abs(ndir[0])
            else:
                i = Index()
                nlen = sqrt(ndir[i]*ndir[i])
            n = ndir / nlen
            r = n

        ufl_assert(r.ufl_shape == o.ufl_shape, "Inconsistent dimensions (in=%d, out=%d)." % (o.ufl_shape[0], r.ufl_shape[0]))
        return r


def change_to_reference_grad(e):
    """Change Grad objects in expression to products of JacobianInverse and ReferenceGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    #return apply_transformer(e, ChangeToReferenceGrad())
    mf = ChangeToReferenceGrad()
    return map_expr_dag(mf, e)


def change_to_reference_geometry(e, physical_coordinates_known, coordinate_coefficient_mapping=None):
    """Change GeometricQuantity objects in expression to the lowest level GeometricQuantity objects.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    mf = ChangeToReferenceGeometry(physical_coordinates_known, coordinate_coefficient_mapping)
    return map_expr_dag(mf, e)


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

    integrand = change_to_reference_grad(integrand)

    integrand = integrand * scale

    if integral_type == "quadrature":
        physical_coordinates_known = True
    else:
        physical_coordinates_known = False
    integrand = change_to_reference_geometry(integrand, physical_coordinates_known)

    return integrand
