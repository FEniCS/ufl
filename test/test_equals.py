"""Test of expression comparison."""

from ufl import Coefficient, Cofunction, FiniteElement, FunctionSpace, Mesh, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1


def test_comparison_of_coefficients():
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    U = FiniteElement("Lagrange", triangle, 2, (), (), "identity", H1)
    Ub = FiniteElement("Lagrange", triangle, 2, (), (), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    u_space = FunctionSpace(domain, U)
    ub_space = FunctionSpace(domain, Ub)

    v1 = Coefficient(v_space, count=1)
    v1b = Coefficient(v_space, count=1)
    v2 = Coefficient(v_space, count=2)
    u1 = Coefficient(u_space, count=1)
    u2 = Coefficient(u_space, count=2)
    u2b = Coefficient(ub_space, count=2)

    # Identical objects
    assert v1 == v1
    assert u2 == u2

    # Equal but distinct objects
    assert v1 == v1b
    assert u2 == u2b

    # Different objects
    assert not v1 == v2
    assert not u1 == u2
    assert not v1 == u1
    assert not v2 == u2


def test_comparison_of_cofunctions():
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    U = FiniteElement("Lagrange", triangle, 2, (), (), "identity", H1)
    Ub = FiniteElement("Lagrange", triangle, 2, (), (), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    u_space = FunctionSpace(domain, U)
    ub_space = FunctionSpace(domain, Ub)

    v1 = Cofunction(v_space.dual(), count=1)
    v1b = Cofunction(v_space.dual(), count=1)
    v2 = Cofunction(v_space.dual(), count=2)
    u1 = Cofunction(u_space.dual(), count=1)
    u2 = Cofunction(u_space.dual(), count=2)
    u2b = Cofunction(ub_space.dual(), count=2)

    # Identical objects
    assert v1 == v1
    assert u2 == u2

    # Equal but distinct objects
    assert v1 == v1b
    assert u2 == u2b

    # Different objects
    assert not v1 == v2
    assert not u1 == u2
    assert not v1 == u1
    assert not v2 == u2


def test_comparison_of_products():
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    v = Coefficient(v_space)
    u = Coefficient(v_space)
    a = (v * 2) * u
    b = (2 * v) * u
    c = 2 * (v * u)
    assert a == b
    assert not a == c
    assert not b == c


def test_comparison_of_sums():
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    v = Coefficient(v_space)
    u = Coefficient(v_space)
    a = (v + 2) + u
    b = (2 + v) + u
    c = 2 + (v + u)
    assert a == b
    assert not a == c
    assert not b == c


def test_comparison_of_deeply_nested_expression():
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    v = Coefficient(v_space, count=1)
    u = Coefficient(v_space, count=1)
    w = Coefficient(v_space, count=2)

    def build_expr(a):
        for i in range(100):
            if i % 3 == 0:
                a = a + i
            elif i % 3 == 1:
                a = a * i
            elif i % 3 == 2:
                a = a ** i
        return a
    a = build_expr(u)
    b = build_expr(v)
    c = build_expr(w)
    assert a == b
    assert not a == c
    assert not b == c
