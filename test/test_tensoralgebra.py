#!/usr/bin/env python

"""
Test tensor algebra operators.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

class TensorAlgebraTestCase(UflTestCase):

    def setUp(self):
        super(TensorAlgebraTestCase, self).setUp()
        self.A = as_matrix([[2, 3], [4, 5]])
        self.B = as_matrix([[6, 7], [8, 9]])
        self.u = as_vector([10, 20])
        self.v = as_vector([30, 40])

    def assertEqualValues(self, A, B):
        B = as_ufl(B)
        self.assertEqual(A.shape(), B.shape())
        self.assertEqual(inner(A-B, A-B)(None), 0)

    def test_repeated_as_tensor(self):
        A = as_tensor(self.A)
        B = as_matrix(self.B)
        u = as_tensor(self.u)
        v = as_vector(self.v)
        self.assertEqual(A, self.A)
        self.assertEqual(B, self.B)
        self.assertEqual(u, self.u)
        self.assertEqual(v, self.v)

    def test_outer(self):
        C = outer(self.u, self.v)
        D = as_matrix([[10*30, 10*40], [20*30, 20*40]])
        self.assertEqualValues(C, D)

        C = outer(self.A, self.v)
        A, v = self.A, self.v
        dims = (0,1)
        D = as_tensor([[[self.A[i,j]*v[k] for k in dims] for j in dims] for i in dims])
        self.assertEqualValues(C, D)

        # TODO: Test other ranks

    def test_inner(self):
        C = inner(self.A, self.B)
        D = 2*6 + 3*7 + 4*8 + 5*9
        self.assertEqualValues(C, D)

        C = inner(self.u, self.v)
        D = 10*30 + 20*40
        self.assertEqualValues(C, D)

    def test_pow2_inner(self):
        f = FacetNormal(triangle)[0]
        f2 = f**2
        self.assertEqual(f2, inner(f, f))

        u2 = self.u**2
        self.assertEqual(u2, inner(self.u, self.u))

        A2 = self.A**2
        self.assertEqual(A2, inner(self.A, self.A))

        # Only tensor**2 notation is supported:
        self.assertRaises(UFLException, lambda: self.A**3)

    def test_dot(self):
        C = dot(self.u, self.v)
        D = 10*30 + 20*40
        self.assertEqualValues(C, D)

        C = dot(self.A, self.B)
        dims = (0,1)
        D = as_matrix([[sum(self.A[i,k]*self.B[k,j] for k in dims) \
                            for j in dims] for i in dims])
        self.assertEqualValues(C, D)

    def test_cross(self):
        u = as_vector([3,3,3])
        v = as_vector([2,2,2])
        C = cross(u, v)
        D = zero(3)
        self.assertEqualValues(C, D)

        u = as_vector([3,3,0])
        v = as_vector([-2,2,0])
        C = cross(u, v)
        z = det(as_matrix([[3,3],[-2,2]]))
        D = as_vector([0,0,z])
        self.assertEqualValues(C, D)

    def xtest_dev(self):
        C = dev(self.A)
        D = 0*C # FIXME: Add expected value here
        self.assertEqualValues(C, D)

    def test_skew(self):
        C = skew(self.A)
        A, dims = self.A, (0,1)
        D = 0.5*as_matrix([[A[i,j] - A[j,i] for j in dims] for i in dims])
        self.assertEqualValues(C, D)

    def test_sym(self):
        C = sym(self.A)
        A, dims = self.A, (0,1)
        D = 0.5*as_matrix([[A[i,j] + A[j,i] for j in dims] for i in dims])
        self.assertEqualValues(C, D)

    def test_transpose(self):
        C = transpose(self.A)
        dims = (0,1)
        D = as_matrix([[self.A[j,i] for j in dims] for i in dims])
        self.assertEqualValues(C, D)

    def test_diag(self):
        dims = (0,1)

        C = diag(self.A)
        D = as_matrix([[(0 if i != j else self.A[i,i]) for j in dims] for i in dims])
        self.assertEqualValues(C, D)

        C = diag(self.u)
        D = as_matrix([[(0 if i != j else self.u[i]) for j in dims] for i in dims])
        self.assertEqualValues(C, D)

    def test_diag_vector(self):
        dims = (0,1)
        C = diag_vector(self.A)
        D = as_vector([self.A[i,i] for i in dims])
        self.assertEqualValues(C, D)

    def test_tr(self):
        C = tr(self.A)
        A, dims = self.A, (0,1)
        D = sum(A[i,i] for i in dims)
        self.assertEqualValues(C, D)

    def xtest_det(self):
        C = det(self.A)
        D = zero() # FIXME: Add expected value here
        self.assertEqualValues(C, D)

    def xtest_cofac(self):
        C = cofac(self.A)
        D = 0*C # FIXME: Add expected value here
        self.assertEqualValues(C, D)

    def xtest_inv(self):
        C = inv(self.A)
        D = 0*C # FIXME: Add expected value here
        self.assertEqualValues(C, D)

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()
