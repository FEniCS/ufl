#!/usr/bin/env python

"""
Tests of the various ways Measure objects can be created and used.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *
#from ufl.domains import as_domain
#from ufl.classes import ...
#from ufl.algorithms import ...

#all_cells = (cell1D, cell2D, cell3D,
#             interval, triangle, tetrahedron,
#             quadrilateral, hexahedron)

from ufl import Measure

class MockMesh:
    def __init__(self, ufl_id):
        self._ufl_id = ufl_id
    def ufl_id(self):
        return self._ufl_id
    def ufl_domain(self):
        return Domain(triangle, 2, 2, "MockMesh_id_%d"%self.ufl_id(), self)
    def ufl_measure(self, domain_type="dx", domain_id="everywhere", metadata=None, domain_data=None):
        return Measure(domain_type, domain_id=domain_id, metadata=metadata, domain=self, domain_data=domain_data)

class MockMeshFunction:
    "Mock class for the pydolfin compatibility hack for domain data with [] syntax."
    def __init__(self, ufl_id, mesh):
        self._mesh = mesh
        self._ufl_id = ufl_id
    def ufl_id(self):
        return self._ufl_id
    def mesh(self):
        return self._mesh
    def ufl_measure(self, domain_type=None, domain_id="everywhere", metadata=None):
        return Measure(domain_type, domain_id=domain_id, metadata=metadata,
                       domain=self.mesh(), domain_data=self)

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
        self.assertEqual(dx.domain_type(), "cell")
        self.assertEqual(dE.domain_type(), "macro_cell")
        #self.assertEqual(dO.domain_type(), "overlap")

        self.assertEqual(ds.domain_type(), "exterior_facet")
        self.assertEqual(dS.domain_type(), "interior_facet")
        self.assertEqual(dc.domain_type(), "surface")
        #self.assertEqual(dI.domain_type(), "interface")

        self.assertEqual(dP.domain_type(), "point")
        #self.assertEqual(dV.domain_type(), "vertex")
        # TODO: Continue this checking

        # Check that defaults are set properly
        self.assertEqual(dx.domain(), None)
        self.assertEqual(dx.metadata(), {})

        # Check that we can create a basic form with default measure
        one = as_ufl(1)
        a = one*dx
        #self.assertEqual(a.domain(), None) # FIXME: This is a key point

    def test_foo(self):

        # Define a manifold domain, allows checking gdim/tdim mixup errors
        gdim = 3
        tdim = 2
        cell = Cell("triangle", gdim)
        mymesh = MockMesh(9)
        mydomain = Domain(cell, gdim, tdim, label="Omega", data=mymesh)

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
                        domain_id=3,
                        metadata=metadata)
        self.assertEqual(mydx.domain().label(), mydomain.label())
        self.assertEqual(mydx.metadata(), metadata)
        M = f*mydx

        # Compatibility:
        dx = Measure("dx")
        #domain=None,
        #domain_id="everywhere",
        #metadata=None)
        self.assertEqual(dx.domain(), None)
        self.assertEqual(dx.domain_id(), "everywhere")

        # Set domain_id to "everywhere", still no domain set
        dxe = dx()
        self.assertEqual(dxe.domain(), None)
        self.assertEqual(dxe.domain_id(), "everywhere")

        # Set domain_id to 5, still no domain set
        dx5 = dx(5)
        self.assertEqual(dx5.domain(), None)
        self.assertEqual(dx5.domain_id(), 5)

        # Check that original dx is untouched
        self.assertEqual(dx.domain(), None)
        self.assertEqual(dx.domain_id(), "everywhere")

        # Set domain_id to (2,3), still no domain set
        dx23 = dx((2,3))
        self.assertEqual(dx23.domain(), None)
        self.assertEqual(dx23.domain_id(), (2,3))

        # Map metadata to metadata, ffc interprets as before
        dxm = dx(metadata={"dummy":123})
        #self.assertEqual(dxm.metadata(), {"dummy":123})
        self.assertEqual(dxm.metadata(), {"dummy":123}) # Deprecated, TODO: Remove

        self.assertEqual(dxm.domain(), None)
        self.assertEqual(dxm.domain_id(), "everywhere")

        #dxm = dx(metadata={"dummy":123})
        #self.assertEqual(dxm.metadata(), {"dummy":123})
        dxm = dx(metadata={"dummy":123})
        self.assertEqual(dxm.metadata(), {"dummy":123})

        self.assertEqual(dxm.domain(), None)
        self.assertEqual(dxm.domain_id(), "everywhere")

        dxi = dx(metadata={"quadrature_degree":3})

        # Mock some dolfin data structures
        dx = Measure("dx")
        ds = Measure("ds")
        dS = Measure("dS")
        mesh = MockMesh(8)
        cell_domains = MockMeshFunction(1, mesh)
        exterior_facet_domains = MockMeshFunction(2, mesh)
        interior_facet_domains = MockMeshFunction(3, mesh)

        self.assertEqual(dx[cell_domains], dx(domain_data=cell_domains))
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
        self.assertTrue(dxd.domain_data() is cell_domains)
        self.assertTrue(dsd.domain_data() is exterior_facet_domains)
        self.assertTrue(dSd.domain_data() is interior_facet_domains)
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
        self.assertEqual(Mx.compute_form_data().subdomain_data[mydomain.label()]["cell"], cell_domains)

        domain, = Ms.domains()
        self.assertEqual(domain.data(), mymesh)
        self.assertEqual(Ms.compute_form_data().subdomain_data[mydomain.label()]["exterior_facet"], exterior_facet_domains)
        
        domain, = MS.domains()
        self.assertEqual(domain.data(), mymesh)
        self.assertEqual(MS.compute_form_data().subdomain_data[mydomain.label()]["interior_facet"], interior_facet_domains)

        # Test joining of these domains in a single form
        domain, = M.domains()
        self.assertEqual(domain.data(), mymesh)
        self.assertEqual(M.compute_form_data().subdomain_data[mydomain.label()]["cell"], cell_domains)
        self.assertEqual(M.compute_form_data().subdomain_data[mydomain.label()]["exterior_facet"], exterior_facet_domains)
        self.assertEqual(M.compute_form_data().subdomain_data[mydomain.label()]["interior_facet"], interior_facet_domains)
        

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()
