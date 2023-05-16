#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest

from ufl import *
from ufl.algorithms.check_arities import ArityMismatch
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1


def test_check_arities():
    # Code from bitbucket issue #49
    cell = tetrahedron
    D = Mesh(cell)
    V = FunctionSpace(D, FiniteElement("Lagrange", cell, 2, (3, ), (3, ), "identity", H1))
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

    fd = compute_form_data(M)
    fd = compute_form_data(L)
    fd = compute_form_data(a)

    assert True


def test_complex_arities():
    cell = tetrahedron
    D = Mesh(cell)
    V = FunctionSpace(D, FiniteElement("Lagrange", cell, 2, (3, ), (3, ), "identity", H1))
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
