#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest

from ufl import *


@pytest.fixture
def element():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    return element


@pytest.fixture
def mass():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    return u * v * dx


@pytest.fixture
def stiffness():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    return inner(grad(u), grad(v)) * dx


@pytest.fixture
def convection():
    cell = triangle
    element = VectorElement("Lagrange", cell, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    w = Coefficient(element)
    return dot(dot(w, nabla_grad(u)), v) * dx


@pytest.fixture
def load():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    f = Coefficient(element)
    v = TestFunction(element)
    return f * v * dx


@pytest.fixture
def boundary_load():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    f = Coefficient(element)
    v = TestFunction(element)
    return f * v * ds


def test_form_arguments(mass, stiffness, convection, load):
    v, u = mass.arguments()
    f, = load.coefficients()

    assert v.number() == 0
    assert u.number() == 1
    assert stiffness.arguments() == (v, u)
    assert load.arguments() == (v,)

    assert (v * dx).arguments() == (v,)
    assert (v * dx + v * ds).arguments() == (v,)
    assert (v * dx + f * v * ds).arguments() == (v,)
    assert (u * v * dx(1) + v * u * dx(2)).arguments() == (v, u)
    assert ((f * v) * u * dx + (u * 3) * (v / 2) * dx(2)).arguments() == (v, u)


def test_form_coefficients(element):
    v = TestFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)

    assert (g * dx).coefficients() == (g,)
    assert (g * dx + g * ds).coefficients() == (g,)
    assert (g * dx + f * ds).coefficients() == (f, g)
    assert (g * dx(1) + f * dx(2)).coefficients() == (f, g)
    assert (g * v * dx + f * v * dx(2)).coefficients() == (f, g)


def test_form_domains():
    cell = triangle
    domain = Domain(cell)
    element = FiniteElement("Lagrange", cell, 1)
    V = FunctionSpace(domain, element)

    v = TestFunction(V)
    f = Coefficient(V)
    x = SpatialCoordinate(domain)[0]

    assert (x * dx).ufl_domains() == (domain,)
    assert (v * dx).ufl_domains() == (domain,)
    assert (f * dx).ufl_domains() == (domain,)
    assert (x * v * f * dx).ufl_domains() == (domain,)
    assert (1 * dx(domain)).ufl_domains() == (domain,)


def test_form_empty(mass):
    assert not mass.empty()
    assert Form([]).empty()


def form_integrals(mass, boundary_load):
    assert isinstance(mass.integrals(), tuple)
    assert len(mass.integrals()) == 1
    assert mass.integrals()[0].integral_type() == "cell"
    assert mass.integrals_by_type("cell") == mass.integrals()
    assert mass.integrals_by_type("exterior_facet") == ()
    assert isinstance(boundary_load.integrals_by_type("cell"), tuple)
    assert len(boundary_load.integrals_by_type("cell")) == 0
    assert len(boundary_load.integrals_by_type("exterior_facet")) == 1
