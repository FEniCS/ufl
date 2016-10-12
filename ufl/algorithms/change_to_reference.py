# -*- coding: utf-8 -*-
"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from ufl.log import error

from ufl.core.multiindex import indices
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag

from ufl.classes import (FormArgument, GeometricQuantity,
                         Terminal, ReferenceGrad, Grad, Restricted, ReferenceValue,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         Indexed, MultiIndex, FixedIndex)

from ufl.constantvalue import as_ufl
from ufl.tensors import as_tensor
from ufl.permutation import compute_indices

from ufl.finiteelement import MixedElement

from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.checks import is_cellwise_constant


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


# FIXME: This implementation semeed to work last year but lead to performance problems. Look through and test again now.
class NEWChangeToReferenceGrad(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        self._ngrads = 0
        self._restricted = ''
        self._avg = ''

    def expr(self, o, *ops):
        return o._ufl_expr_reconstruct_(*ops)

    def terminal(self, o):
        return o

    def coefficient_derivative(self, o, *dummy_ops):
        error("Coefficient derivatives should be expanded before applying change to reference grad.")

    def reference_grad(self, o, *dummy_ops):
        error("Not expecting reference grad while applying change to reference grad.")

    def restricted(self, o, *dummy_ops):
        "Store modifier state."
        if self._restricted != '':
            error("Not expecting nested restrictions.")
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
        if self._avg != '':
            error("Not expecting nested averages.")
        self._avg = "facet"
        f, = o.ufl_operands
        r = self(f)
        self._avg = ""
        return r

    def cell_avg(self, o, *dummy_ops):
        if self._avg != '':
            error("Not expecting nested averages.")
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
        if not isinstance(t, Terminal):
            error("Expecting a Terminal.")

        # Get modifiers accumulated by previous handler calls
        ngrads = self._ngrads
        restricted = self._restricted
        avg = self._avg
        if avg != "":
            error("Averaging not implemented.")  # FIXME

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
        domain = t.ufl_domain()
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
        # rtsh = r.ufl_shape
        gtcomponents = compute_indices(gtsh)
        # rtcomponents = compute_indices(rtsh)

        # Create core modified terminal, with eventual
        # layers of grad applied directly to the terminal,
        # then eventual restriction applied last
        for i in range(ngrads):
            g = Grad(g)
            r = ReferenceGrad(r)
        if restricted:
            g = g(restricted)
            r = r(restricted)

        # Get component indices of global and reference objects with
        # grads applied
        gsh = g.ufl_shape
        # rsh = r.ufl_shape
        # gcomponents = compute_indices(gsh)
        # rcomponents = compute_indices(rsh)

        # Get derivative component indices
        dsh = gsh[len(gtsh):]
        dcomponents = compute_indices(dsh)

        # Create nested array to hold expressions for global
        # components mapped from reference values
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
                # ec, element, eoffset = t.ufl_element().extract_component2(gtc) # FIXME: Translate this correctly
                eoffset = 0
                ec, element = t.ufl_element().extract_reference_component(gtc)

                # Select mapping M from element, pick row emapping =
                # M[ec,:], or emapping = [] if no mapping
                if isinstance(element, MixedElement):
                    error("Expecting a basic element here.")
                mapping = element.mapping()
                if mapping == "contravariant Piola":  # S == HDiv:
                    # Handle HDiv elements with contravariant piola
                    # mapping contravariant_hdiv_mapping = (1/det J) *
                    # J * PullbackOf(o)
                    ec, = ec
                    emapping = Mdiv[ec, :]
                elif mapping == "covariant Piola":  # S == HCurl:
                    # Handle HCurl elements with covariant piola mapping
                    # covariant_hcurl_mapping = JinvT * PullbackOf(o)
                    ec, = ec
                    emapping = K[:, ec]  # Column of K is row of K.T
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
            # if rtsh:
            #     i = Index()
            if len(dsh) != ngrads:
                error("Mismatch between derivative shape and ngrads.")
            if ngrads:
                ii = indices(ngrads)
            else:
                ii = ()

            # Apply mapping row to reference object
            if emapping:  # Mapped, always nonscalar terminal Not
                # using IndexSum for the mapping row dot product to
                # keep it simple, because we don't have a slice type
                emapped_ops = [emapping[s] * Indexed(r, MultiIndex((FixedIndex(eoffset + s),) + ii))
                               for s in range(len(emapping))]
                emapped = sum(emapped_ops[1:], emapped_ops[0])
            elif gtc:  # Nonscalar terminal, unmapped
                emapped = Indexed(r, MultiIndex((FixedIndex(eoffset),) + ii))
            elif ngrads:  # Scalar terminal, unmapped, with derivatives
                emapped = Indexed(r, MultiIndex(ii))
            else:  # Scalar terminal, unmapped, no derivatives
                emapped = r

            for di in dcomponents:
                # Multiply derivative mapping rows, parameterized by
                # free column indices
                dmapping = as_ufl(1)
                for j in range(ngrads):
                    dmapping *= K[ii[j], di[j]]  # Row of K is column of JinvT

                # Compute mapping from reference values for this
                # particular global component
                global_value = dmapping * emapped

                # Apply index sums
                # if rtsh:
                #     global_value = IndexSum(global_value, MultiIndex((i,)))
                # for j in range(ngrads): # Applied implicitly in the dmapping * emapped above
                #     global_value = IndexSum(global_value, MultiIndex((ii[j],)))

                # This is the component index into the full object
                # with grads applied
                gc = gtc + di

                # Insert in nested list
                comp = global_components
                for i in gc[:-1]:
                    comp = comp[i]
                comp[0 if gc == () else gc[-1]] = global_value

        # Wrap nested list in as_tensor unless we have a scalar
        # expression
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
        # Peel off the Grads and count them, and get restriction if
        # it's between the grad and the terminal
        ngrads = 0
        restricted = ''
        rv = False
        while not o._ufl_is_terminal_:
            if isinstance(o, Grad):
                o, = o.ufl_operands
                ngrads += 1
            elif isinstance(o, Restricted):
                restricted = o.side()
                o, = o.ufl_operands
            elif isinstance(o, ReferenceValue):
                rv = True
                o, = o.ufl_operands
            else:
                error("Invalid type %s" % o._ufl_class_.__name__)
        f = o
        if rv:
            f = ReferenceValue(f)

        # Get domain and create Jacobian inverse object
        domain = o.ufl_domain()
        Jinv = JacobianInverse(domain)

        if is_cellwise_constant(Jinv):
            # Optimise slightly by turning Grad(Grad(...)) into
            # J^(-T)J^(-T)RefGrad(RefGrad(...))
            # rather than J^(-T)RefGrad(J^(-T)RefGrad(...))

            # Create some new indices
            ii = indices(len(f.ufl_shape))  # Indices to get to the scalar component of f
            jj = indices(ngrads)  # Indices to sum over the local gradient axes with the inverse Jacobian
            kk = indices(ngrads)  # Indices for the leftover inverse Jacobian axes

            # Preserve restricted property
            if restricted:
                Jinv = Jinv(restricted)
                f = f(restricted)

            # Apply the same number of ReferenceGrad without mappings
            lgrad = f
            for i in range(ngrads):
                lgrad = ReferenceGrad(lgrad)

            # Apply mappings with scalar indexing operations (assumes
            # ReferenceGrad(Jinv) is zero)
            jinv_lgrad_f = lgrad[ii+jj]
            for j, k in zip(jj, kk):
                jinv_lgrad_f = Jinv[j, k]*jinv_lgrad_f

            # Wrap back in tensor shape, derivative axes at the end
            jinv_lgrad_f = as_tensor(jinv_lgrad_f, ii+kk)

        else:
            # J^(-T)RefGrad(J^(-T)RefGrad(...))

            # Preserve restricted property
            if restricted:
                Jinv = Jinv(restricted)
                f = f(restricted)

            jinv_lgrad_f = f
            for foo in range(ngrads):
                ii = indices(len(jinv_lgrad_f.ufl_shape))  # Indices to get to the scalar component of f
                j, k = indices(2)

                lgrad = ReferenceGrad(jinv_lgrad_f)
                jinv_lgrad_f = Jinv[j, k]*lgrad[ii+(j,)]

                # Wrap back in tensor shape, derivative axes at the end
                jinv_lgrad_f = as_tensor(jinv_lgrad_f, ii+(k,))

        return jinv_lgrad_f

    def reference_grad(self, o):
        error("Not expecting reference grad while applying change to reference grad.")

    def coefficient_derivative(self, o):
        error("Coefficient derivatives should be expanded before applying change to reference grad.")


def change_to_reference_grad(e):
    """Change Grad objects in expression to products of JacobianInverse and ReferenceGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    mf = OLDChangeToReferenceGrad()
    # mf = NEWChangeToReferenceGrad()
    return map_expr_dag(mf, e)


def change_integrand_geometry_representation(integrand, scale, integral_type):
    """Change integrand geometry to the right representations."""

    integrand = apply_function_pullbacks(integrand)

    integrand = change_to_reference_grad(integrand)

    integrand = integrand * scale

    if integral_type == "quadrature":
        physical_coordinates_known = True
    else:
        physical_coordinates_known = False
    integrand = apply_geometry_lowering(integrand, physical_coordinates_known)

    return integrand
