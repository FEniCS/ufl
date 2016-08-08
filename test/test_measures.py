#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

"""
Tests of the various ways Measure objects can be created and used.
"""

import pytest

# This imports everything external code will see from ufl
from ufl import *
from ufl.algorithms import compute_form_data

# all_cells = (interval, triangle, tetrahedron,
#             quadrilateral, hexahedron)

from mockobjects import MockMesh, MockMeshFunction


def test_construct_forms_from_default_measures():
    # Create defaults:
    dx = Measure("dx")
    # dE = Measure("dE")

    ds = Measure("ds")
    dS = Measure("dS")

    dP = Measure("dP")
    # dV = Measure("dV")

    dc = Measure("dc")
    # dC = Measure("dC")
    # dO = Measure("dO")
    # dI = Measure("dI")

    ds_b = Measure("ds_b")
    ds_t = Measure("ds_t")
    ds_v = Measure("ds_v")
    dS_h = Measure("dS_h")
    dS_v = Measure("dS_v")

    # Check that names are mapped properly
    assert dx.integral_type() == "cell"
    # assert dE.integral_type() == "macro_cell"

    assert ds.integral_type() == "exterior_facet"
    assert dS.integral_type() == "interior_facet"

    assert dP.integral_type() == "vertex"
    # TODO: Change dP to dV:
    # assert dP.integral_type() == "point"
    # assert dV.integral_type() == "vertex"

    assert dc.integral_type() == "custom"
    assert dC.integral_type() == "cutcell"
    assert dO.integral_type() == "overlap"
    assert dI.integral_type() == "interface"

    # TODO: Remove firedrake hacks:
    assert ds_b.integral_type() == "exterior_facet_bottom"
    assert ds_t.integral_type() == "exterior_facet_top"
    assert ds_v.integral_type() == "exterior_facet_vert"
    assert dS_h.integral_type() == "interior_facet_horiz"
    assert dS_v.integral_type() == "interior_facet_vert"

    # Check that defaults are set properly
    assert dx.ufl_domain() == None
    assert dx.metadata() == {}

    # Check that we can create a basic form with default measure
    one = as_ufl(1)
    a = one * dx(Mesh(triangle))


def test_foo():

    # Define a manifold domain, allows checking gdim/tdim mixup errors
    gdim = 3
    tdim = 2
    cell = Cell("triangle", gdim)
    mymesh = MockMesh(9)
    mydomain = Mesh(cell, ufl_id=9, cargo=mymesh)

    assert cell.topological_dimension() == tdim
    assert cell.geometric_dimension() == gdim
    assert cell.cellname() == "triangle"
    assert mydomain.topological_dimension() == tdim
    assert mydomain.geometric_dimension() == gdim
    assert mydomain.ufl_cell() == cell
    assert mydomain.ufl_id() == 9
    assert mydomain.ufl_cargo() == mymesh

    # Define a coefficient for use in tests below
    V = FunctionSpace(mydomain, FiniteElement("CG", cell, 1))
    f = Coefficient(V)

    # Test definition of a custom measure with explicit parameters
    metadata = {"opt": True}
    mydx = Measure("dx",
                   domain=mydomain,
                   subdomain_id=3,
                   metadata=metadata)
    assert mydx.ufl_domain().ufl_id() == mydomain.ufl_id()
    assert mydx.metadata() == metadata
    M = f * mydx

    # Compatibility:
    dx = Measure("dx")
    # domain=None,
    # subdomain_id="everywhere",
    # metadata=None)
    assert dx.ufl_domain() == None
    assert dx.subdomain_id() == "everywhere"

    # Set subdomain_id to "everywhere", still no domain set
    dxe = dx()
    assert dxe.ufl_domain() == None
    assert dxe.subdomain_id() == "everywhere"

    # Set subdomain_id to 5, still no domain set
    dx5 = dx(5)
    assert dx5.ufl_domain() == None
    assert dx5.subdomain_id() == 5

    # Check that original dx is untouched
    assert dx.ufl_domain() == None
    assert dx.subdomain_id() == "everywhere"

    # Set subdomain_id to (2,3), still no domain set
    dx23 = dx((2, 3))
    assert dx23.ufl_domain() == None
    assert dx23.subdomain_id(), (2 == 3)

    # Map metadata to metadata, ffc interprets as before
    dxm = dx(metadata={"dummy": 123})
    # assert dxm.metadata() == {"dummy":123}
    assert dxm.metadata() == {"dummy": 123}  # Deprecated, TODO: Remove

    assert dxm.ufl_domain() == None
    assert dxm.subdomain_id() == "everywhere"

    # dxm = dx(metadata={"dummy":123})
    # assert dxm.metadata() == {"dummy":123}
    dxm = dx(metadata={"dummy": 123})
    assert dxm.metadata() == {"dummy": 123}

    assert dxm.ufl_domain() == None
    assert dxm.subdomain_id() == "everywhere"

    dxi = dx(metadata={"quadrature_degree": 3})

    # Mock some dolfin data structures
    dx = Measure("dx")
    ds = Measure("ds")
    dS = Measure("dS")
    mesh = MockMesh(8)
    cell_domains = MockMeshFunction(1, mesh)
    exterior_facet_domains = MockMeshFunction(2, mesh)
    interior_facet_domains = MockMeshFunction(3, mesh)

    assert dx[cell_domains] == dx(subdomain_data=cell_domains)
    assert dx[cell_domains] != dx
    assert dx[cell_domains] != dx[exterior_facet_domains]

    # Test definition of a custom measure with legacy bracket syntax
    dxd = dx[cell_domains]
    dsd = ds[exterior_facet_domains]
    dSd = dS[interior_facet_domains]
    # Current behaviour: no domain created, measure domain data is a single
    # object not a full dict
    assert dxd.ufl_domain() == None
    assert dsd.ufl_domain() == None
    assert dSd.ufl_domain() == None
    assert dxd.subdomain_data() is cell_domains
    assert dsd.subdomain_data() is exterior_facet_domains
    assert dSd.subdomain_data() is interior_facet_domains

    # Create some forms with these measures (used in checks below):
    Mx = f * dxd
    Ms = f ** 2 * dsd
    MS = f('+') * dSd
    M = f * dxd + f ** 2 * dsd + f('+') * dSd

    # Test extracting domain data from a form for each measure:
    domain, = Mx.ufl_domains()
    assert domain.ufl_id() == mydomain.ufl_id()
    assert domain.ufl_cargo() == mymesh
    assert Mx.subdomain_data()[mydomain]["cell"] == cell_domains

    domain, = Ms.ufl_domains()
    assert domain.ufl_cargo() == mymesh
    assert Ms.subdomain_data()[mydomain][
        "exterior_facet"] == exterior_facet_domains

    domain, = MS.ufl_domains()
    assert domain.ufl_cargo() == mymesh
    assert MS.subdomain_data()[mydomain][
        "interior_facet"] == interior_facet_domains

    # Test joining of these domains in a single form
    domain, = M.ufl_domains()
    assert domain.ufl_cargo() == mymesh
    assert M.subdomain_data()[mydomain]["cell"] == cell_domains
    assert M.subdomain_data()[mydomain][
        "exterior_facet"] == exterior_facet_domains
    assert M.subdomain_data()[mydomain][
        "interior_facet"] == interior_facet_domains
