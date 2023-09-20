__authors__ = "Marie E. Rognes"

# First added: 2011-11-09
# Last changed: 2011-11-09

from ufl import (Argument, Coefficient, Constant, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction,
                 action, derivative, ds, dS, dx, exp, interval, system)
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1


def test_lhs_rhs_simple():
    V = FiniteElement("Lagrange", interval, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", interval, 1, (1, ), (1, ), "identity", H1))
    space = FunctionSpace(domain, V)
    v = TestFunction(space)
    u = TrialFunction(space)
    w = Argument(space, 2)  # This was 0, not sure why
    f = Coefficient(space)

    F0 = f * u * v * w * dx
    a, L = system(F0)
    assert len(a.integrals()) == 0
    assert len(L.integrals()) == 0

    F1 = derivative(F0, f)
    a, L = system(F1)
    assert len(a.integrals()) == 0
    assert len(L.integrals()) == 0

    F2 = action(F0, f)
    a, L = system(F2)
    assert len(a.integrals()) == 1
    assert len(L.integrals()) == 0

    F3 = action(F2, f)
    a, L = system(F3)
    assert len(L.integrals()) == 1


def test_lhs_rhs_derivatives():
    V = FiniteElement("Lagrange", interval, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", interval, 1, (1, ), (1, ), "identity", H1))
    space = FunctionSpace(domain, V)
    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)

    F0 = exp(f) * u * v * dx + v * dx + f * v * ds + exp(f)('+') * v * dS
    a, L = system(F0)
    assert len(a.integrals()) == 1
    assert len(L.integrals()) == 3

    derivative(F0, f)
    a, L = system(F0)


def test_lhs_rhs_slightly_obscure():
    V = FiniteElement("Lagrange", interval, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", interval, 1, (1, ), (1, ), "identity", H1))
    space = FunctionSpace(domain, V)
    u = TrialFunction(space)
    w = Argument(space, 2)
    f = Constant(domain)

    # FIXME:
    # ufl.algorithsm.formtransformations.compute_form_with_arity
    # is not perfect, e.g. try
    # F = f*u*w*dx + f*w*dx
    F = f * u * w * dx
    a, L = system(F)
    assert len(a.integrals()) == 1
    assert len(L.integrals()) == 0

    F = f * w * dx
    a, L = system(F)
    assert len(L.integrals()) == 1
