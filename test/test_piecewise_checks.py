#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

"""
Test the is_cellwise_constant function on all relevant terminal types.
"""

import pytest
from ufl import *
from ufl.classes import *
from ufl.checks import is_cellwise_constant


def get_domains():
    all_cells = [
        #vertex,
        interval,
        triangle,
        quadrilateral,
        tetrahedron,
        hexahedron,
    ]
    return [Mesh(cell) for cell in all_cells]


def get_nonlinear():
    domains_with_quadratic_coordinates = []
    for D in get_domains():
        V = VectorElement("CG", D.ufl_cell(), 2)
        x = Coefficient(V)
        E = Mesh(x)
        domains_with_quadratic_coordinates.append(E)

    return domains_with_quadratic_coordinates


@pytest.fixture(params=list(range(5)))
def nonlinear_domains(request):
    return get_nonlinear()[request.param]


@pytest.fixture(params=list(range(10)))
def domains_not_linear(request):
    all_domains_not_linear = get_domains() + get_nonlinear()
    return all_domains_not_linear[request.param]


@pytest.fixture(params=list(range(15)))
def domains(request):
    domains = get_domains()
    domains_with_linear_coordinates = []
    for D in domains:
        V = VectorElement("CG", D.ufl_cell(), 1)
        x = Coefficient(V)
        E = Mesh(x)
        domains_with_linear_coordinates.append(E)

    all_domains = domains + domains_with_linear_coordinates + get_nonlinear()
    return all_domains[request.param]


@pytest.fixture(params=list(range(6)))
def affine_domains(request):
    affine_cells = [
        interval,
        triangle,
        tetrahedron,
    ]
    affine_domains = [Mesh(cell) for cell in affine_cells]

    affine_domains_with_linear_coordinates = []
    for D in affine_domains:
        V = VectorElement("CG", D.ufl_cell(), 1)
        x = Coefficient(V)
        E = Mesh(x)
        affine_domains_with_linear_coordinates.append(E)

    all_affine_domains = affine_domains + \
        affine_domains_with_linear_coordinates
    return all_affine_domains[request.param]


@pytest.fixture(params=list(range(8)))
def affine_facet_domains(request):
    affine_facet_cells = [
        interval,
        triangle,
        quadrilateral,
        tetrahedron,
    ]
    affine_facet_domains = [Mesh(cell) for cell in affine_facet_cells]
    affine_facet_domains_with_linear_coordinates = []
    for D in affine_facet_domains:
        V = VectorElement("CG", D.ufl_cell(), 1)
        x = Coefficient(V)
        E = Mesh(x)
        affine_facet_domains_with_linear_coordinates.append(E)

    all_affine_facet_domains = affine_facet_domains + \
        affine_facet_domains_with_linear_coordinates

    return all_affine_facet_domains[request.param]


@pytest.fixture(params=list(range(4)))
def nonaffine_domains(request):
    nonaffine_cells = [
        quadrilateral,
        hexahedron,
    ]
    nonaffine_domains = [Mesh(cell) for cell in nonaffine_cells]
    nonaffine_domains_with_linear_coordinates = []
    for D in nonaffine_domains:
        V = VectorElement("CG", D.ufl_cell(), 1)
        x = Coefficient(V)
        E = Mesh(x)
        nonaffine_domains_with_linear_coordinates.append(E)

    all_nonaffine_domains = nonaffine_domains + \
        nonaffine_domains_with_linear_coordinates

    return all_nonaffine_domains[request.param]


@pytest.fixture(params=list(range(2)))
def nonaffine_facet_domains(request):
    nonaffine_facet_cells = [
        hexahedron,
    ]
    nonaffine_facet_domains = [Mesh(cell) for cell in nonaffine_facet_cells]
    nonaffine_facet_domains_with_linear_coordinates = []
    for D in nonaffine_facet_domains:
        V = VectorElement("CG", D.ufl_cell(), 1)
        x = Coefficient(V)
        E = Mesh(x)
        nonaffine_facet_domains_with_linear_coordinates.append(E)

    all_nonaffine_facet_domains = nonaffine_facet_domains + \
        nonaffine_facet_domains_with_linear_coordinates

    return all_nonaffine_facet_domains[request.param]


def test_always_cellwise_constant_geometric_quantities(domains):
    "Test geometric quantities that are always constant over a cell."
    e = CellVolume(domains)
    assert is_cellwise_constant(e)
    e = Circumradius(domains)
    assert is_cellwise_constant(e)
    e = FacetArea(domains)
    assert is_cellwise_constant(e)
    e = MinFacetEdgeLength(domains)
    assert is_cellwise_constant(e)
    e = MaxFacetEdgeLength(domains)
    assert is_cellwise_constant(e)


def test_coordinates_never_cellwise_constant(domains):
    e = SpatialCoordinate(domains)
    assert not is_cellwise_constant(e)
    e = CellCoordinate(domains)
    assert not is_cellwise_constant(e)


def test_coordinates_never_cellwise_constant_vertex():
    # The only exception here:
    domains = Domain(Cell("vertex", 3))
    assert domains.ufl_cell().cellname() == "vertex"
    e = SpatialCoordinate(domains)
    assert is_cellwise_constant(e)
    e = CellCoordinate(domains)
    assert is_cellwise_constant(e)


def mappings_are_cellwise_constant(domain, test):
    e = Jacobian(domain)
    assert is_cellwise_constant(e) == test
    e = JacobianDeterminant(domain)
    assert is_cellwise_constant(e) == test
    e = JacobianInverse(domain)
    assert is_cellwise_constant(e) == test
    if domain.topological_dimension() != 1:
        e = FacetJacobian(domain)
        assert is_cellwise_constant(e) == test
        e = FacetJacobianDeterminant(domain)
        assert is_cellwise_constant(e) == test
        e = FacetJacobianInverse(domain)
        assert is_cellwise_constant(e) == test


def test_mappings_are_cellwise_constant_on_linear_affine_cells(affine_domains):
    mappings_are_cellwise_constant(affine_domains, True)


def test_mappings_are_cellwise_not_constant_on_nonaffine_cells(nonaffine_domains):
    mappings_are_cellwise_constant(nonaffine_domains, False)


def test_mappings_are_cellwise_not_constant_on_nonlinear_cells(nonlinear_domains):
    mappings_are_cellwise_constant(nonlinear_domains, False)


def facetnormal_cellwise_constant(domain, test):
    e = FacetNormal(domain)
    assert is_cellwise_constant(e) == test


def test_facetnormal_cellwise_constant_affine(affine_facet_domains):
    facetnormal_cellwise_constant(affine_facet_domains, True)


def test_facetnormal_not_cellwise_constant_nonaffine(nonaffine_facet_domains):
    facetnormal_cellwise_constant(nonaffine_facet_domains, False)


def test_facetnormal_not_cellwise_constant_nonlinear(nonlinear_domains):
    facetnormal_cellwise_constant(nonlinear_domains, False)


def test_coefficient_sometimes_cellwise_constant(domains_not_linear):
    e = Constant(domains_not_linear)
    assert is_cellwise_constant(e)

    V = FiniteElement("DG", domains_not_linear.ufl_cell(), 0)
    e = Coefficient(V)
    assert is_cellwise_constant(e)
    V = FiniteElement("R", domains_not_linear.ufl_cell(), 0)
    e = Coefficient(V)
    assert is_cellwise_constant(e)

    # This should be true, but that has to wait for a fix of issue #13
    # e = TestFunction(V)
    # assert is_cellwise_constant(e)
    # V = FiniteElement("R", domains_not_linear.ufl_cell(), 0)
    # e = TestFunction(V)
    # assert is_cellwise_constant(e)


def test_coefficient_mostly_not_cellwise_constant(domains_not_linear):
    V = FiniteElement("DG", domains_not_linear.ufl_cell(), 1)
    e = Coefficient(V)
    assert not is_cellwise_constant(e)
    e = TestFunction(V)
    assert not is_cellwise_constant(e)
