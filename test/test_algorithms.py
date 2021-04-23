# -*- coding: utf-8 -*-

__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-03-12 -- 2009-01-28"

# Modified by Anders Logg, 2008
# Modified by Garth N. Wells, 2009

import pytest
from pprint import *

from ufl import (FiniteElement, TestFunction, TrialFunction, Matrix, triangle,
                 div, grad, Argument, dx, adjoint, Coefficient,
                 FacetNormal, inner, dot, ds)
from ufl.algorithms import (extract_arguments, expand_derivatives,
                            expand_indices, extract_elements,
                            extract_unique_elements, extract_coefficients)
from ufl.corealg.traversal import (pre_traversal, post_traversal,
                                   unique_pre_traversal, unique_post_traversal)

# TODO: add more tests, covering all utility algorithms


@pytest.fixture(scope='module')
def element():
    return FiniteElement("CG", triangle, 1)


@pytest.fixture(scope='module')
def arguments(element):
    v = TestFunction(element)
    u = TrialFunction(element)
    return (v, u)


@pytest.fixture(scope='module')
def coefficients(element):
    c = Coefficient(element)
    f = Coefficient(element)
    return (c, f)


@pytest.fixture
def forms(arguments, coefficients):
    v, u = arguments
    c, f = coefficients
    n = FacetNormal(triangle)
    a = u * v * dx
    L = f * v * dx
    b = u * v * dx(0) + inner(c * grad(u), grad(v)) * \
        dx(1) + dot(n, grad(u)) * v * ds + f * v * dx
    return (a, L, b)


def test_extract_arguments_vs_fixture(arguments, forms):
    assert arguments == tuple(extract_arguments(forms[0]))
    assert tuple(arguments[:1]) == tuple(extract_arguments(forms[1]))


def test_extract_coefficients_vs_fixture(coefficients, forms):
    assert coefficients == tuple(extract_coefficients(forms[2]))


def test_extract_elements_and_extract_unique_elements(forms):
    b = forms[2]
    integrals = b.integrals_by_type("cell")
    integrand = integrals[0].integrand()

    element1 = FiniteElement("CG", triangle, 1)
    element2 = FiniteElement("CG", triangle, 1)

    v = TestFunction(element1)
    u = TrialFunction(element2)

    a = u * v * dx
    assert extract_elements(a) == (element1, element2)
    assert extract_unique_elements(a) == (element1,)


def test_pre_and_post_traversal():
    element = FiniteElement("CG", "triangle", 1)
    v = TestFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)
    p1 = f * v
    p2 = g * v
    s = p1 + p2

    # NB! These traversal algorithms are intended to guarantee only
    # parent before child and vice versa, not this particular
    # ordering:
    assert list(pre_traversal(s)) == [s, p2, g, v, p1, f, v]
    assert list(post_traversal(s)) == [g, v, p2, f, v, p1, s]
    assert list(unique_pre_traversal(s)) == [s, p2, g, v, p1, f]
    assert list(unique_post_traversal(s)) == [v, f, p1, g, p2, s]


def test_expand_indices():
    element = FiniteElement("Lagrange", triangle, 2)
    v = TestFunction(element)
    u = TrialFunction(element)

    def evaluate(form):
        return form.cell_integral()[0].integrand()((), {v: 3, u: 5})  # TODO: How to define values of derivatives?

    a = div(grad(v)) * u * dx
    # a1 = evaluate(a)
    a = expand_derivatives(a)
    # a2 = evaluate(a)
    a = expand_indices(a)
    # a3 = evaluate(a)
    # TODO: Compare a1, a2, a3
    # TODO: Test something more


def test_adjoint():
    cell = triangle

    V1 = FiniteElement("CG", cell, 1)
    V2 = FiniteElement("CG", cell, 2)

    u = TrialFunction(V1)
    v = TestFunction(V2)
    assert u.number() > v.number()

    u2 = Argument(V1, 2)
    v2 = Argument(V2, 3)
    assert u2.number() < v2.number()

    a = u * v * dx
    a_arg_degrees = [arg.ufl_element().degree() for arg in extract_arguments(a)]
    assert a_arg_degrees == [2, 1]

    b = adjoint(a)
    b_arg_degrees = [arg.ufl_element().degree() for arg in extract_arguments(b)]
    assert b_arg_degrees == [1, 2]

    c = adjoint(a, (u2, v2))
    c_arg_degrees = [arg.ufl_element().degree() for arg in extract_arguments(c)]
    assert c_arg_degrees == [1, 2]

    d = adjoint(b)
    d_arg_degrees = [arg.ufl_element().degree() for arg in extract_arguments(d)]
    assert d_arg_degrees == [2, 1]
