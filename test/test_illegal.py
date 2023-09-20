import pytest

from ufl import Argument, Coefficient, FunctionSpace, Mesh, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

# TODO: Add more illegal expressions to check!


@pytest.fixture
def selement():
    return FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)


@pytest.fixture
def velement():
    return FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)


@pytest.fixture
def domain():
    return Mesh(FiniteElement("Lagrange", "triangle", 1, (2, ), (2, ), "identity", H1))


@pytest.fixture
def sspace(domain, selement):
    return FunctionSpace(domain, selement)


@pytest.fixture
def vspace(domain, velement):
    return FunctionSpace(domain, velement)


@pytest.fixture
def a(sspace):
    return Argument(sspace, 2)


@pytest.fixture
def b(sspace):
    return Argument(sspace, 3)


@pytest.fixture
def v(vspace):
    return Argument(vspace, 4)


@pytest.fixture
def u(vspace):
    return Argument(vspace, 5)


@pytest.fixture
def f(sspace):
    return Coefficient(sspace)


@pytest.fixture
def g(sspace):
    return Coefficient(sspace)


@pytest.fixture
def vf(vspace):
    return Coefficient(vspace)


@pytest.fixture
def vg(vspace):
    return Coefficient(vspace)


def test_mul_v_u(v, u):
    with pytest.raises(BaseException):
        v * u


def test_mul_vf_u(vf, u):
    with pytest.raises(BaseException):
        vf * u


def test_mul_vf_vg(vf, vg):
    with pytest.raises(BaseException):
        vf * vg


def test_add_a_v(a, v):
    with pytest.raises(BaseException):
        a + v


def test_add_vf_b(vf, b):
    with pytest.raises(BaseException):
        vf + b


def test_add_vectorexpr_b(vg, v, u, vf, b):
    tmp = vg + v + u + vf
    with pytest.raises(BaseException):
        tmp + b
