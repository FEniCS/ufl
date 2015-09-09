#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
"""
Tests of domain language and attaching domains to forms.
"""

import pytest

from ufl import *
from ufl.domain import as_domain, default_domain
from ufl.algorithms import compute_form_data

all_cells = (interval, triangle, tetrahedron,
             quadrilateral, hexahedron)

from mockobjects import MockMesh, MockMeshFunction


def test_construct_domains_from_cells():
    for cell in all_cells:
        D0 = Mesh(cell)
        D1 = default_domain(cell)
        D2 = as_domain(cell)
        assert D0 is not D1
        assert D0 is not D2
        assert D1 is D2
        if 0:
            print()
            for D in (D1, D2):
                print(('id', id(D)))
                print(('str', str(D)))
                print(('repr', repr(D)))
                print()
        assert D0 != D1
        assert D0 != D2
        assert D1 == D2


def test_as_domain_from_cell_is_equal():
    for cell in all_cells:
        D1 = as_domain(cell)
        D2 = as_domain(cell)
        assert D1 == D2


def test_construct_domains_with_names():
    for cell in all_cells:
        D2 = Mesh(cell, ufl_id=2)
        D3 = Mesh(cell, ufl_id=3)
        D3b = Mesh(cell, ufl_id=3)
        assert D2 != D3
        assert D3 == D3b


def test_domains_sort_by_name():
    # This ordering is rather arbitrary, but at least this shows sorting is
    # working
    domains1 = [Mesh(cell, ufl_id=hash(cell.cellname()))
                for cell in all_cells]
    domains2 = [Mesh(cell, ufl_id=hash(cell.cellname()))
                for cell in sorted(all_cells)]
    sdomains = sorted(domains1, key=lambda D: (D.geometric_dimension(),
                                          D.topological_dimension(),
                                          D.ufl_cell(),
                                          D.ufl_id()))
    assert sdomains != domains1
    assert sdomains == domains2


def test_topdomain_creation():
    D = Mesh(interval)
    assert D.geometric_dimension() == 1
    D = Mesh(triangle)
    assert D.geometric_dimension() == 2
    D = Mesh(tetrahedron)
    assert D.geometric_dimension() == 3


def test_cell_legacy_case():
    # Passing cell like old code does
    D = as_domain(triangle)

    V = FiniteElement("CG", triangle, 1)
    f = Coefficient(V)
    assert f.ufl_domains() == (D, )

    M = f * dx
    assert M.ufl_domains() == (D, )


def test_simple_domain_case():
    # Creating domain from just cell with label like new dolfin will do
    D = Mesh(triangle, ufl_id=3)

    V = FunctionSpace(D, FiniteElement("CG", D.ufl_cell(), 1))
    f = Coefficient(V)
    assert f.ufl_domains() == (D, )

    M = f * dx
    assert M.ufl_domains() == (D, )


def test_creating_domains_with_coordinate_fields(): # FIXME: Rewrite for new approach
    # Definition of higher order domain, element, coefficient, form

    # Mesh with P2 representation of coordinates
    cell = triangle
    P2 = VectorElement("CG", cell, 2)
    domain = Mesh(P2)

    # Piecewise linear function space over quadratic mesh
    element = FiniteElement("CG", cell, 1)
    V = FunctionSpace(domain, element)

    f = Coefficient(V)
    M = f * dx
    assert f.ufl_domains() == (domain, )
    assert M.ufl_domains() == (domain, )

    # Test the gymnastics that dolfin will have to go through
    domain2 = Mesh(P2, ufl_id=domain.ufl_id())
    V2 = FunctionSpace(domain2, eval(repr(V.ufl_element())))
    f2 = Coefficient(V2, count=f.count())
    assert f == f2
    assert domain == domain2
    assert V == V2


def test_join_domains():
    from ufl.domain import join_domains
    mesh7 = MockMesh(7)
    mesh8 = MockMesh(8)
    triangle3 = Cell("triangle", geometric_dimension=3)
    xa = VectorElement("CG", triangle, 1)
    xb = VectorElement("CG", triangle, 1)

    # Equal domains are joined
    assert 1 == len(join_domains([Mesh(triangle, ufl_id=7),
                                  Mesh(triangle, ufl_id=7)]))
    assert 1 == len(join_domains([Mesh(triangle, ufl_id=7, cargo=mesh7),
                                  Mesh(triangle, ufl_id=7, cargo=mesh7)]))
    assert 1 == len(join_domains([Mesh(xa, ufl_id=3), Mesh(xa, ufl_id=3)]))

    # Different domains are not joined
    assert 2 == len(join_domains([Mesh(triangle), Mesh(triangle)]))
    assert 2 == len(join_domains([Mesh(triangle, ufl_id=7),
                                  Mesh(triangle, ufl_id=8)]))
    assert 2 == len(join_domains([Mesh(triangle, ufl_id=7),
                                  Mesh(quadrilateral, ufl_id=8)]))
    assert 2 == len(join_domains([Mesh(xa, ufl_id=7), Mesh(xa, ufl_id=8)]))
    assert 2 == len(join_domains([Mesh(xa), Mesh(xb)]))

    # Incompatible cells require labeling
    # self.assertRaises(UFLException, lambda: join_domains([Mesh(triangle), Mesh(triangle3)]))     # FIXME: Figure out
    # self.assertRaises(UFLException, lambda: join_domains([Mesh(triangle),
    # Mesh(quadrilateral)])) # FIXME: Figure out

    # Incompatible coordinates require labeling
    xc = Coefficient(FunctionSpace(Mesh(triangle), VectorElement("CG", triangle, 1)))
    xd = Coefficient(FunctionSpace(Mesh(triangle), VectorElement("CG", triangle, 1)))
    with pytest.raises(UFLException):
        join_domains([Mesh(xc), Mesh(xd)])

    # Incompatible data is checked if and only if the domains are the same
    assert 2 == len(join_domains([Mesh(triangle, ufl_id=7, cargo=mesh7),
                                  Mesh(triangle, ufl_id=8, cargo=mesh8)]))
    assert 2 == len(join_domains([Mesh(triangle, ufl_id=7, cargo=mesh7),
                                  Mesh(quadrilateral, ufl_id=8, cargo=mesh8)]))
    # Geometric dimensions must match
    with pytest.raises(UFLException):
        join_domains([Mesh(triangle),
                      Mesh(triangle3)])
    with pytest.raises(UFLException):
        join_domains([Mesh(triangle, ufl_id=7, cargo=mesh7),
                      Mesh(triangle3, ufl_id=8, cargo=mesh8)])
    # Cargo and mesh ids must match
    with pytest.raises(UFLException):
        Mesh(triangle, ufl_id=7, cargo=mesh8)

    # Nones are removed
    assert 2 == len(join_domains([None, Mesh(triangle, ufl_id=3),
                                  None, Mesh(triangle, ufl_id=3),
                                  None, Mesh(triangle, ufl_id=4)]))
    assert 2 == len(join_domains([Mesh(triangle, ufl_id=7), None,
                                  Mesh(quadrilateral, ufl_id=8)]))
    assert None not in join_domains([Mesh(triangle3, ufl_id=7), None,
                                     Mesh(tetrahedron, ufl_id=8)])


def test_everywhere_integrals_with_backwards_compatibility():
    D = Mesh(triangle)

    V = FunctionSpace(D, FiniteElement("CG", triangle, 1))
    f = Coefficient(V)

    a = f * dx
    ida, = compute_form_data(a).integral_data

    # Check some integral data
    assert ida.integral_type == "cell"
    assert ida.subdomain_id == "otherwise"
    assert ida.metadata == {}

    # Integrands are not equal because of renumbering
    itg1 = ida.integrals[0].integrand()
    itg2 = a.integrals()[0].integrand()
    assert type(itg1) == type(itg2)
    assert itg1.ufl_element() == itg2.ufl_element()


def xtest_mixed_elements_on_overlapping_regions():  # Old sketch, not working

    # Create domain and both disjoint and overlapping regions
    cell = tetrahedron
    D = Mesh(cell, label='D')
    DD = Region(D, (0, 4), "DD")
    DL = Region(D, (1, 2), "DL")
    DR = Region(D, (2, 3), "DR")

    # Create function spaces on D
    V = FiniteElement("CG", D, 1)
    VD = FiniteElement("DG", DD, 1)
    VC = FiniteElement("R", DD, 0)
    VL = VectorElement("DG", DL, 2)
    VR = FiniteElement("CG", DR, 3)

    # Create mixed space over all these spaces
    M = MixedElement(V, VD, VC, VL, VR)

    # Check that we can get the degree for each value component of the mixed
    # space
    assert M.degree(0) == 1
    assert M.degree(1) == 1
    assert M.degree(2) == 0

    assert M.degree(3) == 2  # Vector element
    assert M.degree(4) == 2
    assert M.degree(5) == 2

    assert M.degree(6) == 3
    assert M.degree() == 3

    # Check that we can get the domain for each value component of the mixed
    # space
    assert M.ufl_domain(0) == D
    assert M.ufl_domain(1) == DD
    assert M.ufl_domain(2) == DD

    assert M.ufl_domain(3) == DL  # Vector element
    assert M.ufl_domain(4) == DL
    assert M.ufl_domain(5) == DL

    assert M.ufl_domain(6) == DR
    # assert M.ufl_domain() == None # FIXME: What?

    # Create a mixed function and fetch components with names for more
    # readable test code below
    m = Coefficient(M)
    md = m[1]
    mc = m[2]
    ml = as_vector((m[3], m[4], m[5]))
    mr = m[6]

    # These should all work out fine with function and integration domains
    # perfectly matching
    a = m[0] ** 2 * dx(D)
    ad = (md ** 2 + mc ** 2) * dx(DD)
    al = ml ** 2 * dx(DL)
    ar = mr ** 2 * dx(DR)

    # TODO: Check properties of forms, maybe by computing and inspecting form
    # data.

    # These should all work out fine with because integration domains are
    # contained in the function domains
    ad = m[0] ** 2 * dx(DD)
    al = m[0] ** 2 * dx(DL)
    ar = m[0] ** 2 * dx("DR")
    a0 = m[0] ** 2 * dx(0)
    a12 = m[0] ** 2 * dx((1, 2))
    a3 = m[0] ** 2 * dx(3)

    # TODO: Check properties of forms, maybe by computing and inspecting form
    # data.

    # These should fail because the functions are undefined on the integration domains
    # self.assertRaises(UFLException, lambda: mr**2*dx(DD)) # FIXME: Make this fail
    # self.assertRaises(UFLException, lambda: mr**2*dx(DD)) # FIXME: Make this
    # fail


def xtest_form_domain_model():  # Old sketch, not working
    # Create domains with different celltypes
    # TODO: Figure out PyDOLFIN integration with Mesh
    DA = Mesh(tetrahedron, label='DA')
    DB = Mesh(hexahedron, label='DB')

    # Check python protocol behaviour
    assert DA != DB
    assert {DA, DA} == {DA}
    assert {DB, DB} == {DB}
    assert {DA, DB} == {DB, DA}
    assert sorted((DA, DB, DA, DB)) == sorted((DB, DA, DA, DB))

    # Check basic properties
    assert DA.name() == 'DA'
    assert DA.geometric_dimension() == 3
    assert DA.topological_dimension() == 3
    assert DA.ufl_cell() == tetrahedron

    # Check region/domain getters
    assert DA.top_domain() == DA
    assert DA.subdomain_ids() == None
    # assert DA.region_names() == []
    # assert DA.regions() == []

    # BDA = Boundary(DA) # TODO
    # IDAB = Intersection(DA,DB) # TODO
    # ODAB = Overlap(DA,DB) # TODO

    # Create overlapping regions of each domain
    DAL = Region(DA, (1, 2), "DAL")
    DAR = Region(DA, (2, 3), "DAR")
    DBL = Region(DB, (0, 1), "DBL")
    DBR = Region(DB, (1, 4), "DBR")

    # Check that regions are available through domains
    # assert DA.region_names() == ['DAL', 'DAR']
    # assert DA.regions() == [DAL, DAR]
    # assert DA["DAR"] == DAR
    # assert DA["DAL"] == DAL

    # Create function spaces on DA
    VA = FiniteElement("CG", DA, 1)
    VAL = FiniteElement("CG", DAL, 1)
    VAR = FiniteElement("CG", DAR, 1)

    # Create function spaces on DB
    VB = FiniteElement("CG", DB, 1)
    VBL = FiniteElement("CG", DBL, 1)
    VBR = FiniteElement("CG", DBR, 1)

    # Check that regions are available through elements
    assert VA.ufl_domain() == DA
    assert VAL.ufl_domain() == DAL
    assert VAR.ufl_domain() == DAR

    # Create functions in each space on DA
    fa = Coefficient(VA)
    fal = Coefficient(VAL)
    far = Coefficient(VAR)

    # Create functions in each space on DB
    fb = Coefficient(VB)
    fbl = Coefficient(VBL)
    fbr = Coefficient(VBR)

    # Checks of coefficient domains is covered well in the mixed element test

    # Create measure objects directly based on domain and region objects
    dxa = dx(DA)
    dxal = dx(DAL)
    dxar = dx('DAR')  # Get Region by name

    # Create measure proxy objects from strings and ints, requiring
    # domains and regions to be part of their integrands
    dxb = dx('DB')   # Get Mesh by name
    dxbl = dx(Region(DB, (1, 4), 'DBL2'))
              # Provide a region with different name but same subdomain ids as
              # DBL
    dxbr = dx((1, 4))
              # Assume unique Mesh and provide subdomain ids explicitly

    # Not checking measure objects in detail, as long as
    # they carry information to construct integrals below
    # they are good to go.

    # Create integrals on each region with matching spaces and measures
    Ma = fa * dxa
    Mar = far * dxar
    Mal = fal * dxal
    Mb = fb * dxb
    Mbr = fbr * dxbr
    Mbl = fbl * dxbl

    # TODO: Check forms, by getting and inspecting domains somehow.

    # TODO: How do we express the distinction between "everywhere" and
    # "nowhere"? subdomain_ids=None vs []?

    # Create forms from integrals over overlapping regions on same top domain
    Marl = Mar + Mal
    Mbrl = Mbr + Mbl

    # Create forms from integrals over top domain and regions
    Mac = Ma + Marl
    Mbc = Mb + Mbrl

    # Create forms from separate top domains
    Mab = Ma + Mb

    # Create forms from separate top domains with overlapping regions
    Mabrl = Mac + Mbc

    # self.assertFalse(True) # Getting here, but it's not bloody likely that
    # everything above is actually working. Add assertions!


def xtest_subdomain_stuff():  # Old sketch, not working
    D = Mesh(triangle)

    D1 = D[1]
    D2 = D[2]
    D3 = D[3]

    DL = Region(D, (D1, D2), 'DL')
    DR = Region(D, (D2, D3), 'DR')
    DM = Overlap(DL, DR)

    assert DM == D2

    VL = VectorElement(DL, "CG", 1)
    VR = FiniteElement(DR, "CG", 2)
    V = VL * VR

    def sub_elements_on_subdomains(W):
        # Get from W: (already there)
        subelements = (VL, VR)
        # Get from W:
        subdomains = (D1, D2, D3)
        # Build in W:
        dom2elm = {D1: (VL,),
                   D2: (VL, VR),
                   D3: (VR,), }
        # Build in W:
        elm2dom = {VL: (D1, D2),
                   VR: (D2, D3)}

    # ElementSwitch represents joining of elements restricted to disjunct
    # subdomains.

    # An element restricted to a domain union becomes a switch
    # of elements restricted to each disjoint subdomain
    VL_D1 = VectorElement(D1, "CG", 1)
    VL_D2 = VectorElement(D2, "CG", 1)
    VLalt = ElementSwitch({D1: VL_D1,
                           D2: VL_D2})
    # Ditto
    VR_D2 = FiniteElement(D2, "CG", 2)
    VR_D3 = FiniteElement(D3, "CG", 2)
    VRalt = ElementSwitch({D2: VR_D2,
                           D3: VR_D3})
    # A mixed element of ElementSwitches is mixed only on the overlapping
    # domains:
    Valt1 = VLalt * VRalt
    Valt2 = ElementSwitch({D1: VL_D1,
                           D2: VL_D2 * VR_D2,
                           D3: VR_D3})

    ul, ur = TrialFunctions(V)
    vl, vr = TestFunctions(V)

    # Implemented by user:
    al = dot(ul, vl) * dx(DL)
    ar = ur * vr * dx(DR)

    # Disjunctified by UFL:
    alonly = dot(ul, vl) * dx(D1)
                 # integral_1 knows that only subelement VL is active
    am = (dot(ul, vl) + ur * vr) * dx(D2)
          # integral_2 knows that both subelements are active
    aronly = ur * vr * \
        dx(D3)  # integral_3 knows that only subelement VR is active
