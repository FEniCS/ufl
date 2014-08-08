#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-12 -- 2009-01-28"

# Modified by Anders Logg, 2008
# Modified by Garth N. Wells, 2009

import pytest
from pprint import *

from ufl import *
from ufl.algorithms import *
from ufl.classes import Sum, Product

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


def test_arguments(arguments, forms):
    assert arguments == tuple(extract_arguments(forms[0]))
    assert tuple(arguments[:1]) == tuple(extract_arguments(forms[1]))


def test_coefficients(coefficients, forms):
    assert coefficients == tuple(extract_coefficients(forms[2]))


def test_elements(forms):
    # print elements(forms[2])
    # print unique_elements(forms[2])
    # print unique_classes(forms[2])
    b = forms[2]
    integrals = b.integrals_by_type(Measure.CELL)
    integrand = integrals[0].integrand()
    d = extract_duplications(integrand)
    # pprint(list(d))

    element1 = FiniteElement("CG", triangle, 1)
    element2 = FiniteElement("CG", triangle, 1)

    v = TestFunction(element1)
    u = TrialFunction(element2)

    a = u * v * dx
    assert (element1, element2) == extract_elements(a)
    assert (element1,) == extract_unique_elements(a)


def test_walk():
    element = FiniteElement("CG", "triangle", 1)
    v = TestFunction(element)
    f = Coefficient(element)
    p = f * v
    a = p * dx

    prestore = []

    def pre(o, stack):
        prestore.append((o, len(stack)))
    poststore = []

    def post(o, stack):
        poststore.append((o, len(stack)))

    for itg in a.integrals_by_type(Measure.CELL):
        walk(itg.integrand(), pre, post)

    assert prestore == [(p, 0), (v, 1), (f, 1)]
                         # NB! Sensitive to ordering of expressions.
    assert poststore == [(v, 1), (f, 1), (p, 0)]
                          # NB! Sensitive to ordering of expressions.
    # print "\n"*2 + "\n".join(map(str,prestore))
    # print "\n"*2 + "\n".join(map(str,poststore))


def test_traversal():
    element = FiniteElement("CG", "triangle", 1)
    v = TestFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)
    p1 = f * v
    p2 = g * v
    s = p1 + p2
    pre_traverse = list(pre_traversal(s))
    post_traverse = list(post_traversal(s))

    assert pre_traverse == [s, p1, v, f, p2, v, g]
        # NB! Sensitive to ordering of expressions.
    assert post_traverse == [v, f, p1, v, g, p2, s]
        # NB! Sensitive to ordering of expressions.


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
    a_arg_degrees = [arg.element().degree() for arg in extract_arguments(a)]
    assert a_arg_degrees == [2, 1]

    b = adjoint(a)
    b_arg_degrees = [arg.element().degree() for arg in extract_arguments(b)]
    assert b_arg_degrees == [1, 2]

    c = adjoint(a, (u2, v2))
    c_arg_degrees = [arg.element().degree() for arg in extract_arguments(c)]
    assert c_arg_degrees == [1, 2]

    d = adjoint(b)
    d_arg_degrees = [arg.element().degree() for arg in extract_arguments(d)]
    assert d_arg_degrees == [2, 1]

if __name__ == "__main__":
    main()
