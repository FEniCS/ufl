"""Test of expression comparison."""

from ufl import Coefficient, Cofunction, FiniteElement, triangle


def test_comparison_of_coefficients():
    V = FiniteElement("CG", triangle, 1)
    U = FiniteElement("CG", triangle, 2)
    Ub = FiniteElement("CG", triangle, 2)
    v1 = Coefficient(V, count=1)
    v1b = Coefficient(V, count=1)
    v2 = Coefficient(V, count=2)
    u1 = Coefficient(U, count=1)
    u2 = Coefficient(U, count=2)
    u2b = Coefficient(Ub, count=2)

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
    V = FiniteElement("CG", triangle, 1)
    U = FiniteElement("CG", triangle, 2)
    Ub = FiniteElement("CG", triangle, 2)
    v1 = Cofunction(V, count=1)
    v1b = Cofunction(V, count=1)
    v2 = Cofunction(V, count=2)
    u1 = Cofunction(U, count=1)
    u2 = Cofunction(U, count=2)
    u2b = Cofunction(Ub, count=2)

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
    V = FiniteElement("CG", triangle, 1)
    v = Coefficient(V)
    u = Coefficient(V)
    a = (v * 2) * u
    b = (2 * v) * u
    c = 2 * (v * u)
    assert a == b
    assert not a == c
    assert not b == c


def test_comparison_of_sums():
    V = FiniteElement("CG", triangle, 1)
    v = Coefficient(V)
    u = Coefficient(V)
    a = (v + 2) + u
    b = (2 + v) + u
    c = 2 + (v + u)
    assert a == b
    assert not a == c
    assert not b == c


def test_comparison_of_deeply_nested_expression():
    V = FiniteElement("CG", triangle, 1)
    v = Coefficient(V, count=1)
    u = Coefficient(V, count=1)
    w = Coefficient(V, count=2)

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
