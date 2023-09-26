import math

from ufl import (Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, VectorConstant, acos, as_tensor, as_ufl,
                 asin, atan, cos, cosh, dx, exp, i, j, ln, max_value, min_value, outer, sin, sinh, tan, tanh, triangle)
from ufl.algorithms import compute_form_data
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1


def xtest_zero_times_argument(self):
    # FIXME: Allow zero forms
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    L = 0*v*dx
    a = 0*(u*v)*dx
    b = (0*u)*v*dx
    assert len(compute_form_data(L).arguments) == 1
    assert len(compute_form_data(a).arguments) == 2
    assert len(compute_form_data(b).arguments) == 2


def test_divisions(self):
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)

    # Test simplification of division by 1
    a = f
    b = f/1
    assert a == b

    # Test simplification of division by 1.0
    a = f
    b = f/1.0
    assert a == b

    # Test simplification of division by of zero by something
    a = 0/f
    b = 0*f
    assert a == b

    # Test simplification of division by self (this simplification has been disabled)
    # a = f/f
    # b = 1
    # assert a == b


def test_products(self):
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    g = Coefficient(space)

    # Test simplification of literal multiplication
    assert f*0 == as_ufl(0)
    assert 0*f == as_ufl(0)
    assert 1*f == f
    assert f*1 == f
    assert as_ufl(2)*as_ufl(3) == as_ufl(6)
    assert as_ufl(2.0)*as_ufl(3.0) == as_ufl(6.0)

    # Test reordering of operands
    assert f*g == g*f

    # Test simplification of self-multiplication (this simplification has been disabled)
    # assert f*f == f**2


def test_sums(self):
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    g = Coefficient(space)

    # Test reordering of operands
    assert f + g == g + f

    # Test adding zero
    assert f + 0 == f
    assert 0 + f == f

    # Test collapsing of basic sum (this simplification has been disabled)
    # assert f + f == 2 * f

    # Test reordering of operands and collapsing sum
    a = f + g + f  # not collapsed, but ordered
    b = g + f + f  # not collapsed, but ordered
    c = (g + f) + f  # not collapsed, but ordered
    d = f + (f + g)  # not collapsed, but ordered
    assert a == b
    assert a == c
    assert a == d

    # Test reordering of operands and collapsing sum
    a = f + f + g  # collapsed
    b = g + (f + f)  # collapsed
    assert a == b


def test_mathfunctions(self):
    for a in (0.1, 0.3, 0.9):
        assert math.sin(a) == sin(a)
        assert math.cos(a) == cos(a)
        assert math.tan(a) == tan(a)
        assert math.sinh(a) == sinh(a)
        assert math.cosh(a) == cosh(a)
        assert math.tanh(a) == tanh(a)
        assert math.asin(a) == asin(a)
        assert math.acos(a) == acos(a)
        assert math.atan(a) == atan(a)
        assert math.exp(a) == exp(a)
        assert math.log(a) == ln(a)
        # TODO: Implement automatic simplification of conditionals?
        assert a == float(max_value(a, a-1))
        # TODO: Implement automatic simplification of conditionals?
        assert a == float(min_value(a, a+1))


def test_indexing(self):
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
    u = VectorConstant(domain)
    v = VectorConstant(domain)

    A = outer(u, v)
    A2 = as_tensor(A[i, j], (i, j))
    assert A2 == A

    Bij = u[i]*v[j]
    Bij2 = as_tensor(Bij, (i, j))[i, j]
    as_tensor(Bij, (i, j))
    assert Bij2 == Bij
