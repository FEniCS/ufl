#!/usr/bin/env py.test

"""
Test tensor algebra operators.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
import pytest

# This imports everything external code will see from ufl
from ufl import *

@pytest.fixture(scope="module")
def A():
    return as_matrix([[2, 3], [4, 5]])

@pytest.fixture(scope="module")
def B():
    return as_matrix([[6, 7], [8, 9]])

@pytest.fixture(scope="module")
def u():
    return as_vector([10, 20])

@pytest.fixture(scope="module")
def v():
    return as_vector([30, 40])


def test_repeated_as_tensor(self, A, B, u, v):
    A2 = as_tensor(A)
    B2 = as_matrix(B)
    u2 = as_tensor(u)
    v2 = as_vector(v)
    assert A2 == A
    assert B2 == B
    assert u2 == u
    assert v2 == v

def test_outer(self, A, B, u, v):
    C = outer(u, v)
    D = as_matrix([[10*30, 10*40], [20*30, 20*40]])
    self.assertEqualValues(C, D)

    C = outer(A, v)
    A, v = A, v
    dims = (0, 1)
    D = as_tensor([[[A[i, j]*v[k] for k in dims] for j in dims] for i in dims])
    self.assertEqualValues(C, D)

    # TODO: Test other ranks

def test_inner(self, A, B, u, v):
    C = inner(A, B)
    D = 2*6 + 3*7 + 4*8 + 5*9
    self.assertEqualValues(C, D)

    C = inner(u, v)
    D = 10*30 + 20*40
    self.assertEqualValues(C, D)

def test_pow2_inner(self, A, u):
    f = FacetNormal(triangle)[0]
    f2 = f*f
    assert f2 == inner(f, f)

    u2 = u**2
    assert u2 == inner(u, u)

    A2 = A**2
    assert A2 == inner(A, A)

    # Only tensor**2 notation is supported:
    self.assertRaises(UFLException, lambda: A**3)

def test_dot(self, A, B, u, v):
    C = dot(u, v)
    D = 10*30 + 20*40
    self.assertEqualValues(C, D)

    C = dot(A, B)
    dims = (0, 1)
    D = as_matrix([[sum(A[i, k]*B[k, j] for k in dims) \
                        for j in dims] for i in dims])
    self.assertEqualValues(C, D)

def test_cross(self):
    u = as_vector([3, 3, 3])
    v = as_vector([2, 2, 2])
    C = cross(u, v)
    D = zero(3)
    self.assertEqualValues(C, D)

    u = as_vector([3, 3, 0])
    v = as_vector([-2, 2, 0])
    C = cross(u, v)
    z = det(as_matrix([[3, 3], [-2, 2]]))
    D = as_vector([0, 0, z])
    self.assertEqualValues(C, D)

def xtest_dev(self, A):
    C = dev(A)
    D = 0*C # FIXME: Add expected value here
    self.assertEqualValues(C, D)

def test_skew(self, A):
    C = skew(A)
    A, dims = A, (0, 1)
    D = 0.5*as_matrix([[A[i, j] - A[j, i] for j in dims] for i in dims])
    self.assertEqualValues(C, D)

def test_sym(self, A):
    C = sym(A)
    A, dims = A, (0, 1)
    D = 0.5*as_matrix([[A[i, j] + A[j, i] for j in dims] for i in dims])
    self.assertEqualValues(C, D)

def test_transpose(self, A):
    C = transpose(A)
    dims = (0, 1)
    D = as_matrix([[A[j, i] for j in dims] for i in dims])
    self.assertEqualValues(C, D)

def test_diag(self, A, u):
    dims = (0, 1)

    C = diag(A)
    D = as_matrix([[(0 if i != j else A[i, i]) for j in dims] for i in dims])
    self.assertEqualValues(C, D)

    C = diag(u)
    D = as_matrix([[(0 if i != j else u[i]) for j in dims] for i in dims])
    self.assertEqualValues(C, D)

def test_diag_vector(self, A):
    dims = (0, 1)
    C = diag_vector(A)
    D = as_vector([A[i, i] for i in dims])
    self.assertEqualValues(C, D)

def test_tr(self, A):
    C = tr(A)
    A, dims = A, (0, 1)
    D = sum(A[i, i] for i in dims)
    self.assertEqualValues(C, D)

def xtest_det(self, A):
    C = det(A)
    D = zero() # FIXME: Add expected value here
    self.assertEqualValues(C, D)

def xtest_cofac(self, A):
    C = cofac(A)
    D = 0*C # FIXME: Add expected value here
    self.assertEqualValues(C, D)

def xtest_inv(self, A):
    C = inv(A)
    D = 0*C # FIXME: Add expected value here
    self.assertEqualValues(C, D)
