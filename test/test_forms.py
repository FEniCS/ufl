#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-12 -- 2008-12-02"

# Modified by Anders Logg, 2008

from ufltestcase import UflTestCase, main

from ufl import *
from ufl.algorithms import *

class MockMesh:
    def __init__(self, ufl_id):
        self._ufl_id = ufl_id
    def ufl_id(self):
        return self._ufl_id
    def ufl_domain(self):
        return Domain(triangle, label="MockMesh_id_%d"%self.ufl_id(), data=self)
    def ufl_measure(self, integral_type="dx", subdomain_id="everywhere", metadata=None, subdomain_data=None):
        return Measure(integral_type, subdomain_id=subdomain_id, metadata=metadata, domain=self, subdomain_data=subdomain_data)

class MockDomainData:
    "Mock class for the pydolfin compatibility hack for domain data with [] syntax."
    def __init__(self, ufl_id):
        self._mesh = MockMesh(10*ufl_id)
        self._ufl_id = ufl_id
    def ufl_id(self):
        return self._ufl_id
    def mesh(self):
        return self._mesh
    def ufl_measure(self, integral_type=None, subdomain_id="everywhere", metadata=None):
        return Measure(integral_type, subdomain_id=subdomain_id, metadata=metadata,
                       domain=self.mesh(), subdomain_data=self)

class TestMeasure(UflTestCase):

    def test_manually_constructing_measures_with_subdomain_data(self):
        # Since we can't write 'dx = dx[data]' in a non-global scope,
        # because of corner cases in the python scope rules,
        # it may be convenient to construct measures directly.
        subdomain_data1 = MockDomainData(1)
        subdomain_data2 = MockDomainData(2)
        subdomain_data3 = MockDomainData(3)

        # Old syntax
        dx = Measure('dx')[subdomain_data1]
        ds = Measure('ds')[subdomain_data2]
        dS = Measure('dS')[subdomain_data3]
        self.assertEqual(dx.subdomain_data(), subdomain_data1)
        self.assertEqual(ds.subdomain_data(), subdomain_data2)
        self.assertEqual(dS.subdomain_data(), subdomain_data3)
        self.assertIsNone(dx.domain())
        self.assertIsNone(ds.domain())
        self.assertIsNone(dS.domain())

        # New syntax (just deprecating [])
        dx = Measure('dx', subdomain_data=subdomain_data1)
        ds = Measure('ds', subdomain_data=subdomain_data2)
        dS = Measure('dS', subdomain_data=subdomain_data3)
        self.assertEqual(dx.subdomain_data(), subdomain_data1)
        self.assertEqual(ds.subdomain_data(), subdomain_data2)
        self.assertEqual(dS.subdomain_data(), subdomain_data3)
        self.assertIsNone(dx.domain())
        self.assertIsNone(ds.domain())
        self.assertIsNone(dS.domain())

        # Mock-up example of how dolfin can simplify measure construction
        dx = subdomain_data1.ufl_measure('dx')
        ds = subdomain_data2.ufl_measure('ds')
        dS = subdomain_data3.ufl_measure('dS')
        self.assertEqual(dx.subdomain_data(), subdomain_data1)
        self.assertEqual(ds.subdomain_data(), subdomain_data2)
        self.assertEqual(dS.subdomain_data(), subdomain_data3)
        self.assertEqual(dx.domain().data(), subdomain_data1.mesh())
        self.assertEqual(ds.domain().data(), subdomain_data2.mesh())
        self.assertEqual(dS.domain().data(), subdomain_data3.mesh())

    def test_functionals_with_metadata(self):
        x, y, z = SpatialCoordinate(tetrahedron)

        a0 = x*dx(0)             + y*dx(0)             + z*dx(1)
        a1 = x*dx(0, {'k': 'v'}) + y*dx(0, {'k': 'v'}) + z*dx(1, {'k': 'v'})
        a2 = x*dx(0, {'k': 'x'}) + y*dx(0, {'k': 'y'}) + z*dx(1, {'k': 'z'})

        b0 = x*dx(0)             + z*dx(1)             + y*dx(0)
        b1 = x*dx(0, {'k': 'v'}) + z*dx(1, {'k': 'v'}) + y*dx(0, {'k': 'v'})
        b2 = x*dx(0, {'k': 'x'}) + z*dx(1, {'k': 'z'}) + y*dx(0, {'k': 'y'})

        c0 = y*dx(0)             + z*dx(1)             + x*dx(0)
        c1 = y*dx(0, {'k': 'v'}) + z*dx(1, {'k': 'v'}) + x*dx(0, {'k': 'v'})
        c2 = y*dx(0, {'k': 'y'}) + z*dx(1, {'k': 'z'}) + x*dx(0, {'k': 'x'})

        d0 = (x*dx(0, {'k': 'xk', 'q':'xq'})
            + y*dx(1, {'k': 'yk', 'q':'yq'}) )
        d1 = (y*dx(1, {'k': 'yk', 'q':'yq'})
            + x*dx(0, {'k': 'xk', 'q':'xq'}))

        a0s = compute_form_data(a0).signature
        a1s = compute_form_data(a1).signature
        a2s = compute_form_data(a2).signature
        b0s = compute_form_data(b0).signature
        b1s = compute_form_data(b1).signature
        b2s = compute_form_data(b2).signature
        c0s = compute_form_data(c0).signature
        c1s = compute_form_data(c1).signature
        c2s = compute_form_data(c2).signature
        d0s = compute_form_data(d0).signature
        d1s = compute_form_data(d1).signature

        # Check stability w.r.t. ordering of terms without compiler data
        self.assertEqual(a0s, b0s)
        self.assertEqual(a0s, c0s)

        # Check stability w.r.t. ordering of terms with equal compiler data
        self.assertEqual(a1s, b1s)
        self.assertEqual(a1s, c1s)

        # Check stability w.r.t. ordering of terms with different compiler data
        self.assertEqual(a2s, b2s)
        self.assertEqual(a2s, c2s)

        # Check stability w.r.t. ordering of terms with two-value compiler data dict
        self.assertEqual(d0s, d1s)

    def test_forms_with_metadata(self):
        element = FiniteElement("Lagrange", triangle, 1)

        u = TrialFunction(element)
        v = TestFunction(element)

        # Three terms on the same subdomain using different representations
        a_0 = (u*v*dx(0, {"representation":"tensor"})
               + inner(grad(u), grad(v))*dx(0, {"representation": "quadrature"})
               + inner(grad(u), grad(v))*dx(0, {"representation": "auto"}))

        # Three terms on different subdomains using different representations and order
        a_1 = (inner(grad(u), grad(v))*dx(0, {"representation":"tensor",
                                              "quadrature_degree":8})
               + inner(grad(u), grad(v))*dx(1, {"representation":"quadrature",
                                                "quadrature_degree":4})
               + inner(grad(u), grad(v))*dx(1, {"representation":"auto",
                                                "quadrature_degree":"auto"}))

        # Sum of the above
        a = a_0 + a_1

        # Same forms with no compiler data:
        b_0 = (u*v*dx(0)
               + inner(grad(u), grad(v))*dx(0)
               + inner(grad(u), grad(v))*dx(0))
        b_1 = (inner(grad(u), grad(v))*dx(0)
               + inner(grad(u), grad(v))*dx(1)
               + inner(grad(u), grad(v))*dx(1))
        b = b_0 + b_1

        # Same forms with same compiler data but different ordering of terms
        c_0 = (inner(grad(u), grad(v))*dx(0, {"representation": "auto"})
               + inner(grad(u), grad(v))*dx(0, {"representation": "quadrature"})
               + u*v*dx(0, {"representation":"tensor"}))
        c_1 = (inner(grad(u), grad(v))*dx(0, {"representation":"tensor",
                                              "quadrature_degree":8})
               + inner(grad(u), grad(v))*dx(1, {"representation":"auto",
                                                "quadrature_degree":"auto"})
               + inner(grad(u), grad(v))*dx(1, {"representation":"quadrature",
                                                "quadrature_degree":4}))
        c = c_0 + c_1

        afd = a.compute_form_data()
        cfd = c.compute_form_data()
        bfd = b.compute_form_data()

        self.assertNotEqual(afd.signature, bfd.signature)
        self.assertEqual(afd.signature, cfd.signature)

    def test_measures_with_subdomain_data_compare_equal_if_subdomain_data_ufl_id_returns_same(self):
        # Configure measure with some arbitrary data object as subdomain_data
        subdomain_data = MockDomainData(3)
        subdomain_data2 = MockDomainData(5)
        dX1 = dx[subdomain_data]
        dX2 = dx(subdomain_data=subdomain_data)
        dX3 = dx[subdomain_data2]
        dX4 = dx(subdomain_data=subdomain_data2)
        self.assertEqual(dX1, dX2)
        self.assertEqual(dX3, dX4)
        self.assertNotEqual(dX1, dX3)
        self.assertNotEqual(dX1, dX4)
        self.assertNotEqual(dX2, dX3)
        self.assertNotEqual(dX2, dX4)
        self.assertNotEqual(dx, dX1)
        self.assertNotEqual(dx, dX2)
        self.assertNotEqual(dx, dX3)
        self.assertNotEqual(dx, dX4)

    # TODO: Move to domains test
    def test_join_domains(self):
        from ufl.geometry import join_domains
        cells = (triangle,)
        for cell in cells:
            m1 = MockMesh(11)
            m2 = MockMesh(22)
            d1 = as_domain(cell)
            d2 = d1.reconstruct(data=m1)
            d3 = d1.reconstruct(data=m2)
            domains2 = join_domains([d1, d2])
            domains3 = join_domains([d1, d3])
            self.assertEqual(len(domains2), 1)
            self.assertEqual(len(domains3), 1)
            self.assertIs(domains2[0].data(), m1)
            self.assertIs(domains3[0].data(), m2)
            self.assertRaises(lambda: join_domains([d2, d3])) # Incompatible data

    def test_measures_with_subdomain_data(self): # FIXME
        # Old code will be using cell instead of domain for a long while
        cell = triangle

        # Define coefficient on this cell
        element = FiniteElement("Lagrange", cell, 1)
        f = Coefficient(element)

        # Check coefficient domain
        domain, = f.domains()
        self.assertEqual(domain.cell(), cell)
        self.assertEqual(domain.data(), None)

        # Configure measure with some arbitrary data object as subdomain_data
        subdomain_data1 = MockDomainData(5)
        subdomain_data2 = MockDomainData(7)
        dX1 = dx(subdomain_data=subdomain_data1)
        dX2 = dx(subdomain_data=subdomain_data2)

        from ufl.common import EmptyDict

        # Check that we get the right subdomain_data from a form
        def _check_form(form, subdomain_data):
            fd = a.compute_form_data()

            # Check form domains and domain data properties
            self.assertEqual(len(form.domains()), 1)
            domain, = form.domains()
            self.assertEqual(domain.cell(), cell)
            self.assertIsNone(domain.data())

            # Repeat checks for preprocessed form
            form = fd.preprocessed_form
            self.assertEqual(len(form.domains()), 1)
            domain, = form.domains()
            data = domain.data()
            self.assertEqual(domain.cell(), cell)
            self.assertIsNone(domain.data())

            if fd.subdomain_data:
                fd_subdomain_data, = fd.subdomain_data.values()
                self.assertIs(fd_subdomain_data.get('cell'), subdomain_data)
                self.assertTrue('exterior_facet' not in fd_subdomain_data)
            else:
                self.assertIsNone(subdomain_data)

        # Build form with no subdomain_data
        a = f*dx + f**2*dx
        _check_form(a, None)

        # Build form with single subdomain_data
        a = f*dX1 + f**2*dX1
        _check_form(a, subdomain_data1)

        a = f*dX2 + f**2*dX2
        _check_form(a, subdomain_data2)

        # Build form with single domain data and domain ids
        a = f*dX1(0) + f**2*dX1(1) + f/3*dX1()
        _check_form(a, subdomain_data1)

        a = f*dX2(0) + f**2*dX2(1) + f/3*dX2()
        _check_form(a, subdomain_data2)

        # Build form from measures with single domain data and no domain data
        a = f*dX1 + f**2*dX1 + f/3*dx
        _check_form(a, subdomain_data1)

        # Build form from measures with single domain data and no domain data, with domain ids
        a = f*dX1(0) + f**2*dX1 + f/3*dx + f/5*dx(2)
        _check_form(a, subdomain_data1)

    def test_integral_data_contains_subdomain_id_otherwise(self):
        # Configure measure with some arbitrary data object as subdomain_data
        #domain = MockDomain(7)
        subdomain_data = MockDomainData(4)
        dX = dx(subdomain_data=subdomain_data)
        #dX2 = dX(domain=domain)

        # Build form with this subdomain_data
        element = FiniteElement("Lagrange", triangle, 1)
        f = Coefficient(element)
        a = f*dX(0) + f**2*dX(1) + f/3*dx

        # Check that integral_data list is consistent as well
        fd = a.compute_form_data()
        f2 = f.reconstruct(count=0)
        for itd in fd.integral_data:
            self.assertEqual(itd.integral_type, 'cell')
            self.assertEqual(itd.metadata, {})
            #self.assertEqual(itd.domain.label(), domain.label())

            if isinstance(itd.subdomain_id, int):
                self.assertEqual(replace(itd.integrals[0].integrand(),
                                         fd.function_replace_map),
                                 f2**(itd.subdomain_id+1) + f2/3)
            else:
                self.assertEqual(itd.subdomain_id, "otherwise")
                self.assertEqual(replace(itd.integrals[0].integrand(),
                                         fd.function_replace_map),
                                 f2/3)

    def test_measures_trigger_exceptions_on_invalid_use(self):
        # Configure measure with some arbitrary data object as subdomain_data
        subdomain_data = MockDomainData(1)
        dX = dx(subdomain_data=subdomain_data)

        # Check that we get an UFL error when using dX without domain id
        #self.assertRaises(UFLException, lambda: f*dX) # This is no longer the case

        # Check that we get a Python error when using unsupported type
        self.assertRaises(TypeError, lambda: "foo"*dX(1))

        # TODO: Document error checks with tests here

    def test_measure_sums(self):
        element = FiniteElement("Lagrange", triangle, 1)
        f = Coefficient(element)

        a1 = f**2*dx(0) + f**2*dx(3)
        a2 = f**2*(dx(0) + dx(3))
        self.assertEqual(a1, a2)

        a3 = f**2*dx(3) + f**2*dx(0)
        a4 = f**2*(dx(3) + dx(0))
        self.assertEqual(a3, a4)

        # Shouldn't we have sorting of integrals?
        #self.assertEqual(a1, a4)

class TestIntegrals(UflTestCase):

    def test_separated_dx(self):
        "Tests automatic summation of integrands over same domain."
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        f = Coefficient(element)
        a = f*v*dx(0) + 2*v*ds + 3*v*dx(0) + 7*v*ds + 3*v*dx(2) + 7*v*dx(2)
        b = (f*v + 3*v)*dx(0) + (2*v + 7*v)*ds + (3*v + 7*v)*dx(2)
        # Check that integrals are represented canonically after preprocessing
        # (these forms have no indices with icky numbering issues)
        afd = a.compute_form_data()
        bfd = b.compute_form_data()
        self.assertEqual(afd.preprocessed_form.integrals(),
                         bfd.preprocessed_form.integrals())
        # And therefore the signatures should be the same
        self.assertEqual(afd.signature, bfd.signature)
        self.assertEqual(compute_form_signature(afd.preprocessed_form), afd.signature)
        self.assertEqual(compute_form_signature(bfd.preprocessed_form), bfd.signature)
        # Note that non-preprocessed signatures are not equal:
        #self.assertEqual(compute_form_signature(a), compute_form_signature(b))

    def test_adding_zero(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        a = v*dx
        b = a + 0
        self.assertEqual(id(a), id(b))
        b = 0 + a
        self.assertEqual(id(a), id(b))
        b = sum([a, 2*a])
        self.assertEqual(b, a+2*a)

class TestFormScaling(UflTestCase):

    def test_scalar_mult_form(self):
        D = Domain(triangle)
        R = FiniteElement("Real", D, 0)
        element = FiniteElement("Lagrange", D, 1)
        v = TestFunction(element)
        f = Coefficient(element)
        c = Coefficient(R)
        # These should be acceptable:
        #self.assertEqual(0*(c*dx), (0*c)*dx) # TODO: Need argument annotation of zero to make this work
        self.assertEqual(0*(c*dx(D)), (0*c)*dx(D))
        self.assertEqual(3*(v*dx), (3*v)*dx)
        self.assertEqual(3.14*(v*dx), (3.14*v)*dx)
        self.assertEqual(c*(v*dx), (c*v)*dx)
        self.assertEqual((c**c+c/3)*(v*dx), ((c**c+c/3)*v)*dx)
        # These should not be acceptable:
        self.assertRaises(TypeError, lambda: f*(v*dx))
        self.assertRaises(TypeError, lambda: (f/2)*(v*dx))
        self.assertRaises(TypeError, lambda: (c*f)*(v*dx))

    def test_action_mult_form(self):
        V = FiniteElement("CG", triangle, 1)
        u = TrialFunction(V)
        v = TrialFunction(V)
        f = Coefficient(V)
        a = u*v*dx
        self.assertEqual(a*f, action(a,f))
        self.assertRaises(TypeError, lambda: a*"foo")


class TestExampleForms(UflTestCase):

    def test_source1(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        f = Coefficient(element)
        a = f*v*dx

    def test_source2(self):
        element = VectorElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        f = Coefficient(element)
        a = dot(f,v)*dx

    def test_source3(self):
        element = TensorElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        f = Coefficient(element)
        a = inner(f,v)*dx

    def test_source4(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        x = SpatialCoordinate(triangle)
        f = sin(x[0])
        a = f*v*dx


    def test_mass1(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = u*v*dx

    def test_mass2(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = u*v*dx

    def test_mass3(self):
        element = VectorElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = dot(u,v)*dx

    def test_mass4(self):
        element = TensorElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(u,v)*dx


    def test_point1(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = u*v*dP(0)


    def test_stiffness1(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = dot(grad(u), grad(v)) * dx

    def test_stiffness2(self):
        element = FiniteElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(grad(u), grad(v)) * dx

    def test_stiffness3(self):
        element = VectorElement("Lagrange", triangle, 1)
        v = TestFunction(element)
        u = TrialFunction(element)
        a = inner(grad(u), grad(v)) * dx


    def test_nonnabla_stiffness_with_conductivity(self):
        velement = VectorElement("Lagrange", triangle, 1)
        telement = TensorElement("Lagrange", triangle, 1)
        v = TestFunction(velement)
        u = TrialFunction(velement)
        M = Coefficient(telement)
        a = inner(grad(u)*M.T, grad(v)) * dx

    def test_nabla_stiffness_with_conductivity(self):
        velement = VectorElement("Lagrange", triangle, 1)
        telement = TensorElement("Lagrange", triangle, 1)
        v = TestFunction(velement)
        u = TrialFunction(velement)
        M = Coefficient(telement)
        a = inner(M*nabla_grad(u), nabla_grad(v)) * dx


    def test_nonnabla_navier_stokes(self):
        cell = triangle
        velement = VectorElement("Lagrange", cell, 2)
        pelement = FiniteElement("Lagrange", cell, 1)
        TH = velement * pelement

        v, q = TestFunctions(TH)
        u, p = TrialFunctions(TH)

        f = Coefficient(velement)
        w = Coefficient(velement)
        Re = Constant(cell)
        dt = Constant(cell)

        a = (dot(u, v) + dt*dot(grad(u)*w, v)
            - dt*Re*inner(grad(u), grad(v))
            + dt*dot(grad(p), v))*dx
        L = dot(f, v)*dx
        b = dot(u, grad(q))*dx

    def test_nabla_navier_stokes(self):
        cell = triangle
        velement = VectorElement("Lagrange", cell, 2)
        pelement = FiniteElement("Lagrange", cell, 1)
        TH = velement * pelement

        v, q = TestFunctions(TH)
        u, p = TrialFunctions(TH)

        f = Coefficient(velement)
        w = Coefficient(velement)
        Re = Constant(cell)
        dt = Constant(cell)

        a = (dot(u, v) + dt*dot(dot(w,nabla_grad(u)), v)
            - dt*Re*inner(grad(u), grad(v))
            + dt*dot(grad(p), v))*dx
        L = dot(f, v)*dx
        b = dot(u, grad(q))*dx


if __name__ == "__main__":
    main()
