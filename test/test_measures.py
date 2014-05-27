#!/use/bin/env python

"""
Tests of the various ways Measure objects can be created and used.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *
from ufl.algorithms import compute_form_data

#all_cells = (cell1D, cell2D, cell3D,
#             interval, triangle, tetrahedron,
#             quadrilateral, hexahedron)

from mockobjects import MockMesh, MockMeshFunction

class MeasureTestCase(UflTestCase):

    def test_construct_forms_from_default_measures(self):
        # Create defaults:
        dx = Measure("dx")
        dE = Measure("dE")
        #dO = Measure("dO")

        ds = Measure("ds")
        dS = Measure("dS")
        dc = Measure("dc")
        #dI = Measure("dI")

        dP = Measure("dP")
        #dV = Measure("dV")

        # Check that names are mapped properly
        self.assertEqual(dx.integral_type(), "cell")
        self.assertEqual(dE.integral_type(), "macro_cell")
        #self.assertEqual(dO.integral_type(), "overlap")

        self.assertEqual(ds.integral_type(), "exterior_facet")
        self.assertEqual(dS.integral_type(), "interior_facet")
        self.assertEqual(dc.integral_type(), "custom")
        #self.assertEqual(dI.integral_type(), "interface")

        self.assertEqual(dP.integral_type(), "point")
        #self.assertEqual(dV.integral_type(), "vertex")
        # TODO: Continue this checking

        # Check that defaults are set properly
        self.assertEqual(dx.domain(), None)
        self.assertEqual(dx.metadata(), {})

        # Check that we can create a basic form with default measure
        one = as_ufl(1)
        a = one*dx(Domain(triangle))

    def test_foo(self):

        # Define a manifold domain, allows checking gdim/tdim mixup errors
        gdim = 3
        tdim = 2
        cell = Cell("triangle", gdim)
        mymesh = MockMesh(9)
        mydomain = Domain(cell, label="Omega", data=mymesh)

        self.assertEqual(cell.topological_dimension(), tdim)
        self.assertEqual(cell.geometric_dimension(), gdim)
        self.assertEqual(cell.cellname(), "triangle")
        self.assertEqual(mydomain.topological_dimension(), tdim)
        self.assertEqual(mydomain.geometric_dimension(), gdim)
        self.assertEqual(mydomain.cell(), cell)
        self.assertEqual(mydomain.label(), "Omega")
        self.assertEqual(mydomain.data(), mymesh)

        # Define a coefficient for use in tests below
        V = FiniteElement("CG", mydomain, 1)
        f = Coefficient(V)

        # Test definition of a custom measure with explicit parameters
        metadata = { "opt": True }
        mydx = Measure("dx",
                        domain=mydomain,
                        subdomain_id=3,
                        metadata=metadata)
        self.assertEqual(mydx.domain().label(), mydomain.label())
        self.assertEqual(mydx.metadata(), metadata)
        M = f*mydx

        # Compatibility:
        dx = Measure("dx")
        #domain=None,
        #subdomain_id="everywhere",
        #metadata=None)
        self.assertEqual(dx.domain(), None)
        self.assertEqual(dx.subdomain_id(), "everywhere")

        # Set subdomain_id to "everywhere", still no domain set
        dxe = dx()
        self.assertEqual(dxe.domain(), None)
        self.assertEqual(dxe.subdomain_id(), "everywhere")

        # Set subdomain_id to 5, still no domain set
        dx5 = dx(5)
        self.assertEqual(dx5.domain(), None)
        self.assertEqual(dx5.subdomain_id(), 5)

        # Check that original dx is untouched
        self.assertEqual(dx.domain(), None)
        self.assertEqual(dx.subdomain_id(), "everywhere")

        # Set subdomain_id to (2,3), still no domain set
        dx23 = dx((2,3))
        self.assertEqual(dx23.domain(), None)
        self.assertEqual(dx23.subdomain_id(), (2,3))

        # Map metadata to metadata, ffc interprets as before
        dxm = dx(metadata={"dummy":123})
        #self.assertEqual(dxm.metadata(), {"dummy":123})
        self.assertEqual(dxm.metadata(), {"dummy":123}) # Deprecated, TODO: Remove

        self.assertEqual(dxm.domain(), None)
        self.assertEqual(dxm.subdomain_id(), "everywhere")

        #dxm = dx(metadata={"dummy":123})
        #self.assertEqual(dxm.metadata(), {"dummy":123})
        dxm = dx(metadata={"dummy":123})
        self.assertEqual(dxm.metadata(), {"dummy":123})

        self.assertEqual(dxm.domain(), None)
        self.assertEqual(dxm.subdomain_id(), "everywhere")

        dxi = dx(metadata={"quadrature_degree":3})

        # Mock some dolfin data structures
        dx = Measure("dx")
        ds = Measure("ds")
        dS = Measure("dS")
        mesh = MockMesh(8)
        cell_domains = MockMeshFunction(1, mesh)
        exterior_facet_domains = MockMeshFunction(2, mesh)
        interior_facet_domains = MockMeshFunction(3, mesh)

        self.assertEqual(dx[cell_domains], dx(subdomain_data=cell_domains))
        self.assertNotEqual(dx[cell_domains], dx)
        self.assertNotEqual(dx[cell_domains], dx[exterior_facet_domains])

        # Test definition of a custom measure with legacy bracket syntax
        dxd = dx[cell_domains]
        dsd = ds[exterior_facet_domains]
        dSd = dS[interior_facet_domains]
        # Current behaviour: no domain created, measure domain data is a single object not a full dict
        self.assertEqual(dxd.domain(), None)
        self.assertEqual(dsd.domain(), None)
        self.assertEqual(dSd.domain(), None)
        self.assertTrue(dxd.subdomain_data() is cell_domains)
        self.assertTrue(dsd.subdomain_data() is exterior_facet_domains)
        self.assertTrue(dSd.subdomain_data() is interior_facet_domains)
        # Considered behaviour at one point:
        #self.assertEqual(dxd.domain().label(), "MockMesh")
        #self.assertEqual(dsd.domain().label(), "MockMesh")
        #self.assertEqual(dSd.domain().label(), "MockMesh")
        #self.assertEqual(dxd.domain().data(),
        #    { "mesh": mesh, "cell": cell_domains })
        #self.assertEqual(dsd.domain().data(),
        #    { "mesh": mesh, "exterior_facet": exterior_facet_domains })
        #self.assertEqual(dSd.domain().data(),
        #    { "mesh": mesh, "interior_facet": interior_facet_domains })

        # Create some forms with these measures (used in checks below):
        Mx = f*dxd
        Ms = f**2*dsd
        MS = f('+')*dSd
        M = f*dxd + f**2*dsd + f('+')*dSd

        # Test extracting domain data from a form for each measure:
        domain, = Mx.domains()
        self.assertEqual(domain.label(), mydomain.label())
        self.assertEqual(domain.data(), mymesh)
        self.assertEqual(Mx.subdomain_data()[mydomain]["cell"], cell_domains)

        domain, = Ms.domains()
        self.assertEqual(domain.data(), mymesh)
        self.assertEqual(Ms.subdomain_data()[mydomain]["exterior_facet"], exterior_facet_domains)

        domain, = MS.domains()
        self.assertEqual(domain.data(), mymesh)
        self.assertEqual(MS.subdomain_data()[mydomain]["interior_facet"], interior_facet_domains)

        # Test joining of these domains in a single form
        domain, = M.domains()
        self.assertEqual(domain.data(), mymesh)
        self.assertEqual(M.subdomain_data()[mydomain]["cell"], cell_domains)
        self.assertEqual(M.subdomain_data()[mydomain]["exterior_facet"], exterior_facet_domains)
        self.assertEqual(M.subdomain_data()[mydomain]["interior_facet"], interior_facet_domains)


# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()
