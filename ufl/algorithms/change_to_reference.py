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

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.classes import (Terminal, ReferenceGrad, Grad,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         FacetJacobian, FacetJacobianInverse, FacetJacobianDeterminant,
                         ReferenceFacetJacobian, QuadratureWeight)
from ufl.constantvalue import as_ufl
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.algorithms.analysis import extract_type
from ufl.indexing import indices
from ufl.tensors import as_tensor
from ufl.compound_expressions import determinant_expr, inverse_expr

class ChangeToReferenceGrad(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)

    def grad(self, o):
        # Peel off the Grads and count them
        ngrads = 0
        while isinstance(o, Grad):
            o, = o.operands()
            ngrads += 1
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

        # Apply the same number of ReferenceGrad without mappings
        lgrad = f
        for i in xrange(ngrads):
            lgrad = ReferenceGrad(lgrad)

        # Apply mappings with scalar indexing operations (assumes ReferenceGrad(Jinv) is zero)
        jinv_lgrad_f = lgrad[ii+jj]
        for j,k in zip(jj,kk):
            jinv_lgrad_f = Jinv[j,k]*jinv_lgrad_f

        # Wrap back in tensor shape, derivative axes at the end
        jinv_lgrad_f = as_tensor(jinv_lgrad_f, ii+kk)

        return jinv_lgrad_f

    def reference_grad(self, o):
        error("Not expecting local grad while applying change to local grad.")

    def coefficient_derivative(self, o):
        error("Coefficient derivatives should be expanded before applying change to local grad.")

class ChangeToReferenceGeometry(ReuseTransformer):
    def __init__(self, physical_coordinates_known):
        ReuseTransformer.__init__(self)
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
            RFJ = ReferenceFacetJacobian(domain)
            i,j,k = indices(3)
            r = as_tensor(J[i,k]*RFJ[k,j], (i,j))
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
                return x

    def reference_coordinate(self, o):
        "Compute from physical coordinates if they are known, using the appropriate mappings."
        if self.physical_coordinates_known:
            r = self._rcache.get(o)
            if r is None:
                K = self.jacobian_inverse(JacobianInverse(domain))
                x = self.spatial_coordinate(SpatialCoordinate(domain))
                x0 = CellOriginCoordinate(domain)
                i,j = indices(2)
                X = as_tensor(K[i,j] * (x[j] - x0[j]), (i,))
                r = X
            return r
        else:
            return o

    def facet_reference_coordinate(self, o):
        if self.physical_coordinates_known:
            error("Missing computation of facet reference coordinates from physical coordinates via mappings.")
        else:
            return o

    def facet_normal(self, o): # FIXE
        r = self._rcache.get(o)
        if r is None:
            domain = o.domain()
            FJ = self.facet_jacobian(FacetJacobian(domain))

            # FIXME: This is 3D only, write up other cases
            tangent0 = tangents[:,0]
            tangent1 = tangents[:,1]
            r = cross(tangent0, tangent1)

            self._rcache[o] = r
        return r


def change_to_reference_grad(e):
    """Change Grad objects in expression to products of JacobianInverse and ReferenceGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    return apply_transformer(e, ChangeToReferenceGrad())


def change_to_reference_geometry(e, physical_coordinates_known):
    """Change Grad objects in expression to products of JacobianInverse and ReferenceGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    return apply_transformer(e, ChangeToReferenceGeometry(physical_coordinates_known))


def compute_integrand_scaling_factor(domain, domain_type):
    """Change integrand geometry to the right representations."""

    weight = QuadratureWeight(domain) # TODO: domain doesn't make any sense here?

    if domain_type == "cell":
        scale = abs(JacobianDeterminant(domain)) * weight
    elif domain_type == "exterior_facet":
        scale = FacetJacobianDeterminant(domain) * weight
    elif domain_type == "interior_facet":
        # TODO: Do we need to restrict here?
        scale = FacetJacobianDeterminant(domain) * weight
    elif domain_type == "quadrature":
        scale = weight
    elif domain_type == "point":
        scale = 1

    return scale


def change_integrand_geometry_representation(integrand, scale, domain_type):
    """Change integrand geometry to the right representations."""

    integrand = change_to_reference_grad(integrand)

    integrand = integrand * scale

    if domain_type == "quadrature":
        physical_coordinates_known = True
    else:
        physical_coordinates_known = False
    integrand = change_to_reference_geometry(integrand, physical_coordinates_known)

    return integrand
