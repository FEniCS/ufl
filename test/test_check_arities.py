#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest
from ufl_legacy import *
from ufl_legacy.algorithms.compute_form_data import compute_form_data
from ufl_legacy.algorithms.check_arities import ArityMismatch


def test_check_arities():
    # Code from bitbucket issue #49
    cell = tetrahedron
    D = Mesh(cell)
    V = FunctionSpace(D, VectorElement("P", cell, 2))
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
    V = FunctionSpace(D, VectorElement("P", cell, 2))
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
