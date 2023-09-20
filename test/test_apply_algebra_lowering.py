import pytest

from ufl import Coefficient, FunctionSpace, Index, Mesh, as_tensor, interval, sqrt, tetrahedron, triangle
from ufl.algorithms.renumbering import renumber_indices
from ufl.compound_expressions import cross_expr, determinant_expr, inverse_expr
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1


@pytest.fixture
def A0(request):
    return Coefficient(FunctionSpace(
        Mesh(FiniteElement("Lagrange", interval, 1, (1, ), (1, ), "identity", H1)),
             FiniteElement("Lagrange", interval, 1, (), (), "identity", H1)))


@pytest.fixture
def A1(request):
    return Coefficient(FunctionSpace(
        Mesh(FiniteElement("Lagrange", interval, 1, (1, ), (1, ), "identity", H1)),
        FiniteElement("Lagrange", interval, 1, (1, 1), (1, 1), "identity", H1)))


@pytest.fixture
def A2(request):
    return Coefficient(FunctionSpace(
        Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)),
        FiniteElement("Lagrange", triangle, 1, (2, 2), (2, 2), "identity", H1)))


@pytest.fixture
def A3(request):
    return Coefficient(FunctionSpace(
        Mesh(FiniteElement("Lagrange", triangle, 1, (3, ), (3, ), "identity", H1)),
        FiniteElement("Lagrange", triangle, 1, (3, 3), (3, 3), "identity", H1)))


@pytest.fixture
def A21(request):
    return Coefficient(FunctionSpace(
        Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)),
        FiniteElement("Lagrange", triangle, 1, (2, 1), (2, 1), "identity", H1)))


@pytest.fixture
def A31(request):
    return Coefficient(FunctionSpace(
        Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)),
        FiniteElement("Lagrange", triangle, 1, (3, 1), (3, 1), "identity", H1)))


@pytest.fixture
def A32(request):
    return Coefficient(FunctionSpace(
        Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)),
        FiniteElement("Lagrange", triangle, 1, (3, 2), (3, 2), "identity", H1)))


def test_determinant0(A0):
    assert determinant_expr(A0) == A0


def test_determinant1(A1):
    assert determinant_expr(A1) == A1[0, 0]


def test_determinant2(A2):
    assert determinant_expr(A2) == A2[0, 0]*A2[1, 1] - A2[0, 1]*A2[1, 0]


def test_determinant3(A3):
    assert determinant_expr(A3) == (A3[0, 0]*(A3[1, 1]*A3[2, 2] - A3[1, 2]*A3[2, 1])
                                    + (A3[1, 0]*A3[2, 2] - A3[1, 2]*A3[2, 0])*(-A3[0, 1])
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
    expected = "TODO"
    assert inverse_expr(A2) == renumber_indices(expected)


def xtest_inverse3(A3):
    expected = "TODO"
    assert inverse_expr(A3) == renumber_indices(expected)


def xtest_pseudo_inverse21(A21):
    expected = "TODO"
    assert renumber_indices(inverse_expr(A21)) == renumber_indices(expected)


def xtest_pseudo_inverse31(A31):
    expected = "TODO"
    assert renumber_indices(inverse_expr(A31)) == renumber_indices(expected)


def xtest_pseudo_inverse32(A32):
    expected = "TODO"
    assert renumber_indices(inverse_expr(A32)) == renumber_indices(expected)
