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

from ufl.classes import (FormArgument, GeometricQuantity,
                         Terminal, ReferenceGrad, Grad, Restricted, ReferenceValue,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         FacetJacobian, FacetJacobianInverse, FacetJacobianDeterminant,
                         CellFacetJacobian,
                         CellEdgeVectors, FacetEdgeVectors,
                         FacetNormal, CellNormal,
                         CellVolume, FacetArea,
                         CellOrientation, FacetOrientation, QuadratureWeight,
                         Indexed, MultiIndex, FixedIndex)

from ufl.finiteelement import MixedElement

from ufl.constantvalue import as_ufl
from ufl.tensoralgebra import Transposed
from ufl.tensors import as_tensor, as_vector
from ufl.operators import sqrt, max_value, min_value
from ufl.permutation import compute_indices

from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.compound_expressions import determinant_expr, cross_expr, inverse_expr
from ufl.finiteelement import FiniteElement, EnrichedElement, VectorElement, MixedElement, OuterProductElement


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

        if isinstance(element, (FiniteElement, EnrichedElement, OuterProductElement)):
            mapping = element.mapping()
            if mapping == "identity":
                global_value = local_value
            elif mapping == "contravariant Piola":
                # contravariant_hdiv_mapping = (1/det J) * J * PullbackOf(o)
                J = Jacobian(o.domain())
                detJ = JacobianDeterminant(o.domain())
                mapping = (1/detJ) * J
                i, j = indices(2)
                global_value = as_vector(mapping[i, j] * local_value[j], i)
            elif mapping == "covariant Piola":
                # covariant_hcurl_mapping = JinvT * PullbackOf(o)
                Jinv = JacobianInverse(o.domain())
                i, j = indices(2)
                JinvT = as_tensor(Jinv[i, j], (j, i))
                mapping = JinvT
                global_value = as_vector(mapping[i, j] * local_value[j], i)
            else:
                error("Mapping type %s not handled" % mapping)
        elif isinstance(element, VectorElement):
            # Allow VectorElement of CG/DG (scalar-valued), throw error
            # on anything else (can be supported at a later date, if needed)
            mapping = element.mapping()
            if mapping == "identity" and len(o.element().value_shape()) == 1:
                global_value = local_value
        elif isinstance(element, MixedElement):
            error("Mixed Functions must be split")
        else:
            error("Unknown element %s", str(element))

        return global_value

    form_coefficient = form_argument


# FIXME: This implementation semeed to work last year but lead to performance problems. Look through and test again now.
class NEWChangeToReferenceGrad(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        self._ngrads = 0
        self._restricted = ''
        self._avg = ''

    def expr(self, o, *ops):
        return o.reconstruct(*ops)

    def terminal(self, o):
        return o

    def coefficient_derivative(self, o, *dummy_ops):
        error("Coefficient derivatives should be expanded before applying change to reference grad.")

    def reference_grad(self, o, *dummy_ops):
        error("Not expecting reference grad while applying change to reference grad.")

    def restricted(self, o, *dummy_ops):
        "Store modifier state."
        ufl_assert(self._restricted == '', "Not expecting nested restrictions.")
        self._restricted = o.side()
        f, = o.ufl_operands
        r = self(f)
        self._restricted = ''
        return r

    def grad(self, o, *dummy_ops):
        "Store modifier state."
        self._ngrads += 1
        f, = o.ufl_operands
        r = self(f)
        self._ngrads -= 1
        return r

    def facet_avg(self, o, *dummy_ops):
        ufl_assert(self._avg == '', "Not expecting nested averages.")
        self._avg = "facet"
        f, = o.ufl_operands
        r = self(f)
        self._avg = ""
        return r

    def cell_avg(self, o, *dummy_ops):
        ufl_assert(self._avg == '', "Not expecting nested averages.")
        self._avg = "cell"
        f, = o.ufl_operands
        r = self(f)
        self._avg = ""
        return r

    def form_argument(self, t):
        return self._mapped(t)

    def geometric_quantity(self, t):
        if self._restricted or self._ngrads or self._avg:
            return self._mapped(t)
        else:
            return t

    def _mapped(self, t):
        # Check that we have a valid input object
        ufl_assert(isinstance(t, Terminal), "Expecting a Terminal.")

        # Get modifiers accumulated by previous handler calls
        ngrads = self._ngrads
        restricted = self._restricted
        avg = self._avg
        ufl_assert(avg == "", "Averaging not implemented.") # FIXME

        # These are the global (g) and reference (r) values
        if isinstance(t, FormArgument):
            g = t
            r = ReferenceValue(g)
        elif isinstance(t, GeometricQuantity):
            g = t
            r = g
        else:
            error("Unexpected type {0}.".format(type(t).__name__))

        # Some geometry mapping objects we may need multiple times below
        domain = t.domain()
        J = Jacobian(domain)
        detJ = JacobianDeterminant(domain)
        K = JacobianInverse(domain)

        # Restrict geometry objects if applicable
        if restricted:
            J = J(restricted)
            detJ = detJ(restricted)
            K = K(restricted)

        # Create Hdiv mapping from possibly restricted geometry objects
        Mdiv = (1.0/detJ) * J

        # Get component indices of global and reference terminal objects
        gtsh = g.ufl_shape
        rtsh = r.ufl_shape
        gtcomponents = compute_indices(gtsh)
        rtcomponents = compute_indices(rtsh)

        # Create core modified terminal, with eventual
        # layers of grad applied directly to the terminal,
        # then eventual restriction applied last
        for i in range(ngrads):
            g = Grad(g)
            r = ReferenceGrad(r)
        if restricted:
            g = g(restricted)
            r = r(restricted)

        # Get component indices of global and reference objects with grads applied
        gsh = g.ufl_shape
        rsh = r.ufl_shape
        gcomponents = compute_indices(gsh)
        rcomponents = compute_indices(rsh)

        # Get derivative component indices
        dsh = gsh[len(gtsh):]
        dcomponents = compute_indices(dsh)

        # Create nested array to hold expressions for global components mapped from reference values
        def ndarray(shape):
            if len(shape) == 0:
                return [None]
            elif len(shape) == 1:
                return [None]*shape[-1]
            else:
                return [ndarray(shape[1:]) for i in range(shape[0])]
        global_components = ndarray(gsh)

        # Compute mapping from reference values for each global component
        for gtc in gtcomponents:

            if isinstance(t, FormArgument):

                # Find basic subelement and element-local component
                #ec, element, eoffset = t.element().extract_component2(gtc) # FIXME: Translate this correctly
                eoffset = 0
                ec, element = t.element().extract_reference_component(gtc)

                # Select mapping M from element, pick row emapping = M[ec,:], or emapping = [] if no mapping
                ufl_assert(not isinstance(element, MixedElement),
                           "Expecting a basic element here.")
                mapping = element.mapping()
                if mapping == "contravariant Piola": #S == HDiv:
                    # Handle HDiv elements with contravariant piola mapping
                    # contravariant_hdiv_mapping = (1/det J) * J * PullbackOf(o)
                    ec, = ec
                    emapping = Mdiv[ec,:]
                elif mapping == "covariant Piola": #S == HCurl:
                    # Handle HCurl elements with covariant piola mapping
                    # covariant_hcurl_mapping = JinvT * PullbackOf(o)
                    ec, = ec
                    emapping = K[:,ec] # Column of K is row of K.T
                elif mapping == "identity":
                    emapping = None
                else:
                    error("Unknown mapping {0}".format(mapping))

            elif isinstance(t, GeometricQuantity):
                eoffset = 0
                emapping = None

            else:
                error("Unexpected type {0}.".format(type(t).__name__))

            # Create indices
            #if rtsh:
            #    i = Index()
            ufl_assert(len(dsh) == ngrads, "Mismatch between derivative shape and ngrads.")
            if ngrads:
                ii = indices(ngrads)
            else:
                ii = ()

            # Apply mapping row to reference object
            if emapping: # Mapped, always nonscalar terminal
                # Not using IndexSum for the mapping row dot product to keep it simple,
                # because we don't have a slice type
                emapped_ops = [emapping[s] * Indexed(r, MultiIndex((FixedIndex(eoffset + s),) + ii))
                               for s in range(len(emapping))]
                emapped = sum(emapped_ops[1:], emapped_ops[0])
            elif gtc: # Nonscalar terminal, unmapped
                emapped = Indexed(r, MultiIndex((FixedIndex(eoffset),) + ii))
            elif ngrads: # Scalar terminal, unmapped, with derivatives
                emapped = Indexed(r, MultiIndex(ii))
            else: # Scalar terminal, unmapped, no derivatives
                emapped = r

            for di in dcomponents:
                # Multiply derivative mapping rows, parameterized by free column indices
                dmapping = as_ufl(1)
                for j in range(ngrads):
                    dmapping *= K[ii[j], di[j]] # Row of K is column of JinvT

                # Compute mapping from reference values for this particular global component
                global_value = dmapping * emapped

                # Apply index sums
                #if rtsh:
                #    global_value = IndexSum(global_value, MultiIndex((i,)))
                #for j in range(ngrads): # Applied implicitly in the dmapping * emapped above
                #    global_value = IndexSum(global_value, MultiIndex((ii[j],)))

                # This is the component index into the full object with grads applied
                gc = gtc + di

                # Insert in nested list
                comp = global_components
                for i in gc[:-1]:
                    comp = comp[i]
                comp[0 if gc == () else gc[-1]] = global_value

        # Wrap nested list in as_tensor unless we have a scalar expression
        if gsh:
            tensor = as_tensor(global_components)
        else:
            tensor, = global_components
        return tensor


class OLDChangeToReferenceGrad(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, o):
        return o

    def grad(self, o):
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

    def reference_grad(self, o):
        error("Not expecting reference grad while applying change to reference grad.")

    def coefficient_derivative(self, o):
        error("Coefficient derivatives should be expanded before applying change to reference grad.")


class ChangeToReferenceGeometry(MultiFunction):
    def __init__(self, physical_coordinates_known, coordinate_coefficient_mapping):
        MultiFunction.__init__(self)
        self.coordinate_coefficient_mapping = coordinate_coefficient_mapping or {}
        self.physical_coordinates_known = physical_coordinates_known
        self._cache = {} # Needed by memoized_handler

    expr = MultiFunction.reuse_if_untouched

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
    def _future_jacobian(self, o):
        # If we're not using Coefficient to represent the spatial coordinate,
        # we can just as well just return o here too unless we add representation
        # of basis functions and dofs to the ufl layer (which is nice to avoid).
        return o

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
    def _future_spatial_coordinate(self, o):
        "Fall through to coordinate field of domain if it exists."
        if self.physical_coordinates_known:
            return o
        else:
            # If we're not using Coefficient to represent the spatial coordinate,
            # we can just as well just return o here too unless we add representation
            # of basis functions and dofs to the ufl layer (which is nice to avoid).
            return o

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


def change_to_reference_value(e):
    """Change coefficients and arguments in expression to apply Piola mappings

    @param e:
        An Expr or Form.
    """
    return apply_transformer(e, ChangeToReferenceValue())


def change_to_reference_grad(e):
    """Change Grad objects in expression to products of JacobianInverse and ReferenceGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    mf = OLDChangeToReferenceGrad()
    #mf = NEWChangeToReferenceGrad()
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

    elif integral_type.startswith("exterior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant and quadrature weight
            scale = FacetJacobianDeterminant(domain) * weight
        else:
            # No need to scale 'integral' over a vertex
            scale = 1

    elif integral_type.startswith("interior_facet"):
        if tdim > 1:
            # Scaling integral by facet jacobian determinant from one side and quadrature weight
            scale = FacetJacobianDeterminant(domain)('+') * weight
        else:
            # No need to scale 'integral' over a vertex
            scale = 1

    elif integral_type in ("custom", "interface", "overlap", "cutcell"):
        # Scaling with custom weight, which includes eventual volume scaling
        scale = weight

    elif integral_type in ("vertex", "point"):
        # No need to scale 'integral' over a point
        scale = 1

    else:
        error("Unknown integral type {}, don't know how to scale.".format(integral_type))

    return scale


def change_integrand_geometry_representation(integrand, scale, integral_type):
    """Change integrand geometry to the right representations."""

    integrand = change_to_reference_grad(integrand)

    integrand = change_to_reference_value(integrand)

    integrand = integrand * scale

    if integral_type == "quadrature":
        physical_coordinates_known = True
    else:
        physical_coordinates_known = False
    integrand = change_to_reference_geometry(integrand, physical_coordinates_known)

    return integrand
