__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-03-12 -- 2009-01-28"

# Modified by Anders Logg, 2008
# Modified by Garth N. Wells, 2009

import pytest

from ufl import (Argument, Coefficient, FacetNormal, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction,
                 VectorElement, adjoint, div, dot, ds, dx, grad, inner, triangle)
from ufl.algorithms import (expand_derivatives, expand_indices, extract_arguments, extract_coefficients,
                            extract_elements, extract_unique_elements)
from ufl.corealg.traversal import post_traversal, pre_traversal, unique_post_traversal, unique_pre_traversal

# TODO: add more tests, covering all utility algorithms


@pytest.fixture(scope='module')
def element():
    return FiniteElement("CG", triangle, 1)


@pytest.fixture(scope='module')
def domain():
    return Mesh(VectorElement("CG", triangle, 1))


@pytest.fixture(scope='module')
def space(element, domain):
    return FunctionSpace(domain, element)


@pytest.fixture(scope='module')
def arguments(space):
    v = TestFunction(space)
    u = TrialFunction(space)
    return (v, u)


@pytest.fixture(scope='module')
def coefficients(space):
    c = Coefficient(space)
    f = Coefficient(space)
    return (c, f)


@pytest.fixture
def forms(arguments, coefficients, space):
    v, u = arguments
    c, f = coefficients
    n = FacetNormal(space)
    a = u * v * dx
    L = f * v * dx
    b = u * v * dx(0) + inner(c * grad(u), grad(v)) * dx(1) + dot(n, grad(u)) * v * ds + f * v * dx
    return (a, L, b)


def test_extract_arguments_vs_fixture(arguments, forms):
    assert arguments == tuple(extract_arguments(forms[0]))
    assert tuple(arguments[:1]) == tuple(extract_arguments(forms[1]))


def test_extract_coefficients_vs_fixture(coefficients, forms):
    assert coefficients == tuple(extract_coefficients(forms[2]))


def test_extract_elements_and_extract_unique_elements(forms, domain):
    b = forms[2]
    integrals = b.integrals_by_type("cell")
    integrals[0].integrand()

    element1 = FiniteElement("CG", triangle, 1)
    element2 = FiniteElement("CG", triangle, 1)

    space1 = FunctionSpace(domain, element1)
    space2 = FunctionSpace(domain, element2)

    v = TestFunction(space1)
    u = TrialFunction(space2)

    a = u * v * dx
    assert extract_elements(a) == (element1, element2)
    assert extract_unique_elements(a) == (element1,)


def test_pre_and_post_traversal(domain):
    element = FiniteElement("CG", "triangle", 1)
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)
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


def test_expand_indices(domain):
    element = FiniteElement("Lagrange", triangle, 2)
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)

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


def test_adjoint(domain):
    cell = triangle

    V1 = FiniteElement("CG", cell, 1)
    V2 = FiniteElement("CG", cell, 2)

    s1 = FunctionSpace(domain, V1)
    s2 = FunctionSpace(domain, V2)

    u = TrialFunction(s1)
    v = TestFunction(s2)
    assert u.number() > v.number()

    u2 = Argument(s1, 2)
    v2 = Argument(s2, 3)
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
