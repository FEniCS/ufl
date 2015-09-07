#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest
from ufl import *
from ufl.algorithms.compute_form_data import compute_form_data

def test_check_arities():
    # Code from bitbucket issue #49
    cell = tetrahedron
    D = Domain(cell)
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
