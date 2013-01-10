#!/usr/bin/env python

"""
Tests of domain language and attaching domains to forms.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *
from ufl.domains import as_domain
#from ufl.classes import ...
#from ufl.algorithms import ...

all_cells = (cell1D, cell2D, cell3D,
             interval, triangle, tetrahedron,
             quadrilateral, hexahedron)

class DomainTestCase(UflTestCase):

    def test_construct_domains_from_cells(self):
        for cell in all_cells:
            D1 = Domain(cell)
            D2 = as_domain(cell)
            self.assertFalse(D1 is D2)
            self.assertEqual(D1, D2)

    def test_as_domain_from_cell_is_unique(self):
        for cell in all_cells:
            D1 = as_domain(cell)
            D2 = as_domain(cell)
            self.assertTrue(D1 is D2)

    def test_construct_domains_with_names(self):
        for cell in all_cells:
            D2 = Domain(cell, name="D2")
            D3 = Domain(cell, name="D3")
            self.assertNotEqual(D2, D3)

    def test_domains_sort_by_name(self):
        domains = [Domain(cell, "D%d"%k) for (k,cell) in enumerate(all_cells)]
        #sdomains = sorted(domains) # FIXME
        #self.assertEqual()

    def test_extract_domains_from_form(self):
        cell = triangle
        # FIXME

    def test_(self):
        cell = triangle
        # FIXME

    def test_topdomain_creation(self):
        D = Domain(triangle)

    def test_numbered_subdomains_are_registered(self):
        D = Domain(triangle)

        D1 = D[1]
        D2 = D[2]

        subdomain_ids = D.disjoint_subdomain_ids()

        self.assertEqual(subdomain_ids, [1, 2])

    def test_named_subdomain_groups_are_registered(self):
        D = Domain(triangle)

        D1 = D[1]
        D2 = D[2]
        D3 = D[3]

        DL = DomainGroup(D, (D1, D2), 'DL')
        DR = DomainGroup(D, (D2, D3), 'DR')

        subdomain_groups = D.subdomain_groups()

        self.assertEqual(subdomain_groups, [DL, DR])

    def xtest_subdomain_stuff(self):
        D = Domain(triangle)

        D1 = D[1]
        D2 = D[2]
        D3 = D[3]

        DL = DomainGroup(D, (D1, D2), 'DL')
        DR = DomainGroup(D, (D2, D3), 'DR')
        DM = Overlap(DL, DR)

        self.assertEqual(DM, D2)

        VL = VectorElement(DL, "CG", 1)
        VR = FiniteElement(DR, "CG", 2)
        V = VL*VR

        def sub_elements_on_subdomains(W):
            # Get from W: (already there)
            subelements = (VL, VR)
            # Get from W:
            subdomains = (D1, D2, D3)
            # Build in W:
            dom2elm = { D1: (VL,),
                        D2: (VL,VR),
                        D3: (VR,), }
            # Build in W:
            elm2dom = { VL: (D1,D2),
                        VR: (D2,D3) }

        # ElementSwitch represents joining of elements restricted to disjunct subdomains.

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
        # A mixed element of ElementSwitches is mixed only on the overlapping domains:
        Valt1 = VLalt*VRalt
        Valt2 = ElementSwitch({D1: VL_D1,
                               D2: VL_D2*VR_D2,
                               D3: VR_D3})

        ul, ur = TrialFunctions(V)
        vl, vr = TestFunctions(V)

        # Implemented by user:
        al = dot(ul,vl)*dx(DL)
        ar = ur*vr*dx(DR)

        # Disjunctified by UFL:
        alonly = dot(ul,vl)*dx(D1) # integral_1 knows that only subelement VL is active
        am = (dot(ul,vl) + ur*vr)*dx(D2) # integral_2 knows that both subelements are active
        aronly = ur*vr*dx(D3) # integral_3 knows that only subelement VR is active

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()
