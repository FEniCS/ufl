"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.checks import is_cellwise_constant
from ufl.classes import Grad, JacobianInverse, ReferenceGrad, ReferenceValue, Restricted
from ufl.core.multiindex import indices
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.tensors import as_tensor

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
e0 = || J[:,:] . <-1, 1> || = || < J[0,1]-J[0,0], J[1,1]-J[1,0] > || = || dx/dX <-1,1> ||
    = edge length of edge 0 (v1-v2)

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


class ChangeToReferenceGrad(MultiFunction):
    """Change to reference grad."""

    def __init__(self):
        """Initalise."""
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, o):
        """Apply to terminal."""
        return o

    def grad(self, o):
        """Apply to grad."""
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
                raise ValueError(f"Invalid type {o._ufl_class_.__name__}")
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
            jinv_lgrad_f = lgrad[ii + jj]
            for j, k in zip(jj, kk):
                jinv_lgrad_f = Jinv[j, k] * jinv_lgrad_f

            # Wrap back in tensor shape, derivative axes at the end
            jinv_lgrad_f = as_tensor(jinv_lgrad_f, ii + kk)

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
                jinv_lgrad_f = Jinv[j, k] * lgrad[ii + (j,)]

                # Wrap back in tensor shape, derivative axes at the end
                jinv_lgrad_f = as_tensor(jinv_lgrad_f, ii + (k,))

        return jinv_lgrad_f

    def reference_grad(self, o):
        """Apply to reference_grad."""
        raise ValueError("Not expecting reference grad while applying change to reference grad.")

    def coefficient_derivative(self, o):
        """Apply to coefficient_derivative."""
        raise ValueError("Coefficient derivatives should be expanded before applying change to reference grad.")


def change_to_reference_grad(e):
    """Change Grad objects in expression to products of JacobianInverse and ReferenceGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    Args:
        e: An Expr or Form.
    """
    mf = ChangeToReferenceGrad()
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
