# -*- coding: utf-8 -*-

import pytest
from ufl_legacy import *
from ufl_legacy.compound_expressions import *
from ufl_legacy.algorithms.renumbering import renumber_indices


@pytest.fixture
def A0(request):
    return Coefficient(FiniteElement("CG", interval, 1))


@pytest.fixture
def A1(request):
    return Coefficient(TensorElement("CG", interval, 1))


@pytest.fixture
def A2(request):
    return Coefficient(TensorElement("CG", triangle, 1))


@pytest.fixture
def A3(request):
    return Coefficient(TensorElement("CG", tetrahedron, 1))


@pytest.fixture
def A21(request):
    return Coefficient(TensorElement("CG", triangle, 1, shape=(2, 1)))


@pytest.fixture
def A31(request):
    return Coefficient(TensorElement("CG", triangle, 1, shape=(3, 1)))


@pytest.fixture
def A32(request):
    return Coefficient(TensorElement("CG", triangle, 1, shape=(3, 2)))


def test_determinant0(A0):
    assert determinant_expr(A0) == A0


def test_determinant1(A1):
    assert determinant_expr(A1) == A1[0, 0]


def test_determinant2(A2):
    assert determinant_expr(A2) == A2[0, 0]*A2[1, 1] - A2[0, 1]*A2[1, 0]


def test_determinant3(A3):
    assert determinant_expr(A3) == (A3[0, 0]*(A3[1, 1]*A3[2, 2] - A3[1, 2]*A3[2, 1])
                                    + A3[0, 1]*(A3[1, 2]*A3[2, 0] - A3[1, 0]*A3[2, 2])
                                    + A3[0, 2]*(A3[1, 0]*A3[2, 1] - A3[1, 1]*A3[2, 0]))


def test_pseudo_determinant21(A21):
    i = Index()
    assert renumber_indices(determinant_expr(A21)) == renumber_indices(sqrt(A21[i, 0]*A21[i, 0]))


def test_pseudo_determinant31(A31):
    i = Index()
    assert renumber_indices(determinant_expr(A31)) == renumber_indices(sqrt((A31[i, 0]*A31[i, 0])))


def test_pseudo_determinant32(A32):
    i = Index()
    c = cross_expr(A32[:, 0], A32[:, 1])
    assert renumber_indices(determinant_expr(A32)) == renumber_indices(sqrt(c[i]*c[i]))


def test_inverse0(A0):
    expected = 1.0/A0  # stays scalar
    assert inverse_expr(A0) == renumber_indices(expected)


def test_inverse1(A1):
    expected = as_tensor(((1.0/A1[0, 0],),))  # reshaped into 1x1 tensor
    assert inverse_expr(A1) == renumber_indices(expected)


def xtest_inverse2(A2):
    expected = todo
    assert inverse_expr(A2) == renumber_indices(expected)


def xtest_inverse3(A3):
    expected = todo
    assert inverse_expr(A3) == renumber_indices(expected)


def xtest_pseudo_inverse21(A21):
    expected = todo
    assert renumber_indices(inverse_expr(A21)) == renumber_indices(expected)


def xtest_pseudo_inverse31(A31):
    expected = todo
    assert renumber_indices(inverse_expr(A31)) == renumber_indices(expected)


def xtest_pseudo_inverse32(A32):
    expected = todo
    assert renumber_indices(inverse_expr(A32)) == renumber_indices(expected)
