#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
"""
Tests of domain language and attaching domains to forms.
"""

import pytest

from ufl import *
from ufl.geometry import as_domain
from ufl.algorithms import compute_form_data

all_cells = (interval, triangle, tetrahedron,
             quadrilateral, hexahedron)

from mockobjects import MockMesh, MockMeshFunction


def test_construct_domains_from_cells():
    for cell in all_cells:
        D1 = Domain(cell)
        D2 = as_domain(cell)
        assert D1 is not D2
        if 0:
            print()
            for D in (D1, D2):
                print(('id', id(D)))
                print(('str', str(D)))
                print(('repr', repr(D)))
                print()
        assert D1 == D2


def test_as_domain_from_cell_is_equal():
    for cell in all_cells:
        D1 = as_domain(cell)
        D2 = as_domain(cell)
        assert D1 == D2


def test_construct_domains_with_names():
    for cell in all_cells:
        D2 = Domain(cell, label="D2")
        D3 = Domain(cell, label="D3")
        D3b = Domain(cell, label="D3")
        assert D2 != D3
        assert D3 == D3b


def test_domains_sort_by_name():
    # This ordering is rather arbitrary, but at least this shows sorting is
    # working
    domains1 = [Domain(cell, label="D%s" % cell.cellname())
                for cell in all_cells]
    domains2 = [Domain(cell, label="D%s" % cell.cellname())
                for cell in sorted(all_cells)]
    sdomains = sorted(domains1, key=lambda D: (D.geometric_dimension(),
                                               D.topological_dimension(),
                      D.ufl_cell(),
                      D.ufl_label()))
    assert sdomains != domains1
    assert sdomains == domains2


def test_topdomain_creation():
    D = Domain(interval)
    assert D.geometric_dimension() == 1
    D = Domain(triangle)
    assert D.geometric_dimension() == 2
    D = Domain(tetrahedron)
    assert D.geometric_dimension() == 3


def test_cell_legacy_case():
    # Passing cell like old code does
    D = as_domain(triangle)

    V = FiniteElement("CG", triangle, 1)
    f = Coefficient(V)
    assert f.domains() == (D, )

    M = f * dx
    assert M.domains() == (D, )


def test_simple_domain_case():
    # Creating domain from just cell with label like new dolfin will do
    D = Domain(triangle, label="foo")

    V = FiniteElement("CG", D, 1)
    f = Coefficient(V)
    assert f.domains() == (D, )

    M = f * dx
    assert M.domains() == (D, )


def test_creating_domains_with_coordinate_fields():
    # P2 field for coordinates
    D = Domain(triangle)
    P2 = VectorElement("CG", D, 2)
    x = Coefficient(P2)
    assert x.domains() == (D, )

    # Definition of higher order domain, element, coefficient, form
    E = Domain(x)
    V = FiniteElement("CG", E, 1)
    f = Coefficient(V)
    M = f * dx
    assert f.domains() == (E, )
    assert M.domains() == (E, )

    # Test the gymnastics that dolfin will have to go through
    V2 = eval(V.reconstruction_signature())
    E2 = V2.domain().reconstruct(coordinates=x)
    V2 = V2.reconstruct(domain=E2)
    f2 = f.reconstruct(element=V2)
    assert f == f2
    assert V == V2
    assert E == E2


def test_join_domains():
    from ufl.geometry import join_domains
    mesh1 = MockMesh(11)
    mesh2 = MockMesh(13)
    triangle3 = Cell("triangle", geometric_dimension=3)
    xa = Coefficient(VectorElement("CG", Domain(triangle, label="A"), 1))
    xb = Coefficient(VectorElement("CG", Domain(triangle, label="B"), 1))

    # Equal domains are joined
    assert 1 == len(join_domains([Domain(triangle), Domain(triangle)]))
    assert 1 == len(join_domains([Domain(triangle, label="A"),
                                  Domain(triangle, label="A")]))
    assert 1 == len(join_domains([Domain(triangle, label="A", data=mesh1),
                                  Domain(triangle, label="A", data=mesh1)]))
    assert 1 == len(join_domains([Domain(xa), Domain(xa)]))

    # Different domains are not joined
    assert 2 == len(join_domains([Domain(triangle, label="A"),
                                  Domain(triangle, label="B")]))
    assert 2 == len(join_domains([Domain(triangle, label="A"),
                                  Domain(quadrilateral, label="B")]))
    assert 2 == len(join_domains([Domain(xa),
                                  Domain(xb)]))

    # Incompatible cells require labeling
    # self.assertRaises(UFLException, lambda: join_domains([Domain(triangle), Domain(triangle3)]))     # FIXME: Figure out
    # self.assertRaises(UFLException, lambda: join_domains([Domain(triangle),
    # Domain(quadrilateral)])) # FIXME: Figure out

    # Incompatible coordinates require labeling
    xc = Coefficient(VectorElement("CG", Domain(triangle), 1))
    xd = Coefficient(VectorElement("CG", Domain(triangle), 1))
    with pytest.raises(UFLException):
        join_domains([Domain(xc), Domain(xd)])

    # Incompatible data is checked if and only if the domains are the same
    assert 2 == len(join_domains([Domain(triangle, label="A", data=mesh1),
                                  Domain(triangle, label="B", data=mesh2)]))
    assert 2 == len(join_domains([Domain(triangle, label="A", data=mesh1),
                                  Domain(triangle3, label="B", data=mesh2)]))
    assert 2 == len(join_domains([Domain(triangle, label="A", data=mesh1),
                                  Domain(quadrilateral, label="B", data=mesh2)]))
    with pytest.raises(UFLException):
        join_domains([Domain(triangle, label="A", data=mesh1),
                      Domain(triangle, label="A", data=mesh2)])

    # Nones are removed
    assert 1 == len(
        join_domains([None, Domain(triangle), None, Domain(triangle), None]))
    assert 2 == len(join_domains([Domain(triangle, label="A"), None,
                                  Domain(quadrilateral, label="B")]))
    assert None not in join_domains([Domain(triangle, label="A"), None,
                                     Domain(tetrahedron, label="B")])


def test_everywhere_integrals_with_backwards_compatibility():
    D = Domain(triangle)

    V = FiniteElement("CG", D, 1)
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


def xtest_mixed_elements_on_overlapping_regions():
    # Create domain and both disjoint and overlapping regions
    D = Domain(tetrahedron, label='D')
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
    assert M.domain(0) == D
    assert M.domain(1) == DD
    assert M.domain(2) == DD

    assert M.domain(3) == DL  # Vector element
    assert M.domain(4) == DL
    assert M.domain(5) == DL

    assert M.domain(6) == DR
    # assert M.domain() == None # FIXME: What?

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


def xtest_form_domain_model():
    # Create domains with different celltypes
    # TODO: Figure out PyDOLFIN integration with Domain
    DA = Domain(tetrahedron, label='DA')
    DB = Domain(hexahedron, label='DB')

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
    assert VA.domain() == DA
    assert VAL.domain() == DAL
    assert VAR.domain() == DAR

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
    dxb = dx('DB')   # Get Domain by name
    dxbl = dx(Region(DB, (1, 4), 'DBL2'))
              # Provide a region with different name but same subdomain ids as
              # DBL
    dxbr = dx((1, 4))
              # Assume unique Domain and provide subdomain ids explicitly

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
    D = Domain(triangle)

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
