#!/usr/bin/env python

"""
Tests of the change to local representaiton algorithms.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

from ufl import *
from ufl.classes import LocalGrad, JacobianInverse
from ufl.algorithms import tree_format, change_to_local_grad, renumber_indices

class TestChangeToLocal(UflTestCase):
    def test_change_to_local_grad(self):
        domain = Domain(triangle)
        U = FiniteElement("CG", domain, 1)
        V = VectorElement("CG", domain, 1)
        u = Coefficient(U)
        v = Coefficient(V)
        Jinv = JacobianInverse(domain)
        i,j,k = indices(3)
        q,r,s = indices(3)
        t, = indices(1)

        # Single grad change on a scalar function
        expr = grad(u)
        actual = change_to_local_grad(expr)
        expected = as_tensor(Jinv[k,i]*LocalGrad(u)[k], (i,))
        self.assertEqual(renumber_indices(actual), renumber_indices(expected))

        # Single grad change on a vector valued function
        expr = grad(v)
        actual = change_to_local_grad(expr)
        expected = as_tensor(Jinv[k,j]*LocalGrad(v)[i,k], (i,j))
        self.assertEqual(renumber_indices(actual), renumber_indices(expected))

        # Multiple grads should work fine for affine domains:
        expr = grad(grad(u))
        actual = change_to_local_grad(expr)
        expected = as_tensor(Jinv[s,j]*(Jinv[r,i]*LocalGrad(LocalGrad(u))[r,s]), (i,j))
        self.assertEqual(renumber_indices(actual), renumber_indices(expected))

        expr = grad(grad(grad(u)))
        actual = change_to_local_grad(expr)
        expected = as_tensor(Jinv[s,k]*(Jinv[r,j]*(Jinv[q,i]*LocalGrad(LocalGrad(LocalGrad(u)))[q,r,s])), (i,j,k))
        self.assertEqual(renumber_indices(actual), renumber_indices(expected))

        # Multiple grads on a vector valued function
        expr = grad(grad(v))
        actual = change_to_local_grad(expr)
        expected = as_tensor(Jinv[s,j]*(Jinv[r,i]*LocalGrad(LocalGrad(v))[t,r,s]), (t,i,j))
        self.assertEqual(renumber_indices(actual), renumber_indices(expected))

        expr = grad(grad(grad(v)))
        actual = change_to_local_grad(expr)
        expected = as_tensor(Jinv[s,k]*(Jinv[r,j]*(Jinv[q,i]*LocalGrad(LocalGrad(LocalGrad(v)))[t,q,r,s])), (t,i,j,k))
        self.assertEqual(renumber_indices(actual), renumber_indices(expected))

        #print tree_format(expected)
        #print tree_format(actual)
        #print tree_format(renumber_indices(actual))
        #print tree_format(renumber_indices(expected))

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

