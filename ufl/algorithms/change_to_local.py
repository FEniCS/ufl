"""Algorithm for replacing gradients in an expression with local gradients and coordinate mappings."""

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
from ufl.classes import Terminal, LocalGrad, Grad, JacobianInverse
from ufl.constantvalue import as_ufl
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.algorithms.analysis import extract_type
from ufl.indexing import indices
from ufl.tensors import as_tensor

class ChangeToLocalGrad(ReuseTransformer):
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

        # Apply the same number of LocalGrad without mappings
        lgrad = f
        for i in xrange(ngrads):
            lgrad = LocalGrad(lgrad)

        # Apply mappings with scalar indexing operations (assumes LocalGrad(Jinv) is zero)
        jinv_lgrad_f = lgrad[ii+jj]
        for j,k in zip(jj,kk):
            jinv_lgrad_f = Jinv[j,k]*jinv_lgrad_f

        # Wrap back in tensor shape, derivative axes at the end
        jinv_lgrad_f = as_tensor(jinv_lgrad_f, ii+kk)

        return jinv_lgrad_f

    def local_grad(self, o):
        error("Not expecting local grad while applying change to local grad.")

    def coefficient_derivative(self, o):
        error("Coefficient derivatives should be expanded before applying change to local grad.")

def change_to_local_grad(e):
    """Change Grad objects in expression to products of JacobianInverse and LocalGrad.

    Assumes the expression is preprocessed or at least that derivatives have been expanded.

    @param e:
        An Expr or Form.
    """
    return apply_transformer(e, ChangeToLocalGrad())
