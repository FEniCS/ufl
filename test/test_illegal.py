import pytest

from ufl import *
from ufl.algorithms import *

# TODO: Add more illegal expressions to check!


def selement():
    return FiniteElement("Lagrange", "triangle", 1)


def velement():
    return VectorElement("Lagrange", "triangle", 1)


@pytest.fixture
def a():
    return Argument(selement(), 2)


@pytest.fixture
def b():
    return Argument(selement(), 3)


@pytest.fixture
def v():
    return Argument(velement(), 4)


@pytest.fixture
def u():
    return Argument(velement(), 5)


@pytest.fixture
def f():
    return Coefficient(selement())


@pytest.fixture
def g():
    return Coefficient(selement())


@pytest.fixture
def vf():
    return Coefficient(velement())


@pytest.fixture
def vg():
    return Coefficient(velement())


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
