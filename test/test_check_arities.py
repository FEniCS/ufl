import pytest

from ufl import (Coefficient, FacetNormal, FunctionSpace, Mesh, SpatialCoordinate, TestFunction, TrialFunction, adjoint,
                 cofac, conj, derivative, ds, dx, grad, inner, tetrahedron)
from ufl.algorithms.check_arities import ArityMismatch
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.finiteelement import FiniteElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1


def test_check_arities():
    # Code from bitbucket issue #49
    cell = tetrahedron
    D = Mesh(FiniteElement("Lagrange", cell, 1, (3, ), (3, ), identity_pull_back, H1))
    V = FunctionSpace(D, FiniteElement("Lagrange", cell, 2, (3, ), (3, ), identity_pull_back, H1))
    dv = TestFunction(V)
    du = TrialFunction(V)

    X = SpatialCoordinate(D)
    N = FacetNormal(D)

    u = Coefficient(V)
    x = X + u
    F = grad(x)
    n = cofac(F) * N

    M = inner(x, n) * ds
    L = derivative(M, u, dv)
    a = derivative(L, u, du)

    compute_form_data(M)
    compute_form_data(L)
    compute_form_data(a)


def test_complex_arities():
    cell = tetrahedron
    D = Mesh(FiniteElement("Lagrange", cell, 1, (3, ), (3, ), identity_pull_back, H1))
    V = FunctionSpace(D, FiniteElement("Lagrange", cell, 2, (3, ), (3, ), identity_pull_back, H1))
    v = TestFunction(V)
    u = TrialFunction(V)

    # Valid form.
    F = inner(u, v) * dx
    compute_form_data(F, complex_mode=True)
    # Check that adjoint conjugates correctly
    compute_form_data(adjoint(F), complex_mode=True)

    with pytest.raises(ArityMismatch):
        compute_form_data(inner(v, u) * dx, complex_mode=True)

    with pytest.raises(ArityMismatch):
        compute_form_data(inner(conj(v), u) * dx, complex_mode=True)
