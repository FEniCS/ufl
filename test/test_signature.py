#!/usr/bin/env python

"""
Test the computation of form signatures.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

from ufl.common import EmptyDictType
from ufl.classes import MultiIndex
from ufl.algorithms.signature import compute_multiindex_hashdata, \
    compute_terminal_hashdata, compute_form_signature

from itertools import chain

# TODO: Test compute_terminal_hashdata
#   TODO: Check that form argument counts only affect the sig by their relative ordering
#   TODO: Check that all other relevant terminal propeties affect the terminal_hashdata

# TODO: Test that operator types affect the sig
# TODO: Test that we do not get collisions for some large sets of generated forms
# TODO: How do we know that we have tested the signature reliably enough?

from ufl.algorithms import compute_form_signature

def domain_numbering2(*domains):
    items = []
    for i, domain in enumerate(domains):
        key = (domain.cell(), domain.label())
        items.append((key, i))
    return dict(items)

def domain_numbering(*cells):
    return domain_numbering2(*[as_domain(c) for c in cells])

class TerminalHashDataTestCase(UflTestCase):

    def test_domain_signatures_of_cell2domains(self):
        all_cells = (interval, quadrilateral, hexahedron, triangle, tetrahedron, cell1D, cell2D, cell3D)
        for cell in all_cells:
            # Equality holds when constructing two domains from a cell:
            self.assertEqual(as_domain(cell), as_domain(cell))
            # Hash value holds when constructing two domains from a cell:
            self.assertEqual(hash(as_domain(cell)), hash(as_domain(cell)))
            # Signature data holds when constructing two domains from a cell:
            self.assertEqual(as_domain(cell).signature_data(domain_numbering(cell)),
                             as_domain(cell).signature_data(domain_numbering(cell)))

    def compute_unique_terminal_hashdatas(self, hashdatas):
        count = 0
        data = set()
        hashes = set()
        reprs = set()
        for d in hashdatas:
            # Each d is the result of a compute_terminal_hashdatas call,
            # which is a dict where the keys are non-canonical terminals
            # and the values are the canonical hashdata.
            # We want to count unique hashdata values,
            # ignoring the original terminals.
            assert isinstance(d, dict)
            # Sorting values by hash should be stable at least in a single test run:
            t = tuple(sorted(list(d.values()), key=lambda x: hash(x)))
            #print t

            # Add the hashdata values tuple to sets based on itself, its hash,
            # and its repr (not sure why I included repr anymore?)
            hashes.add(hash(t)) # This will fail if t is not hashable, which it should be!
            data.add(t)
            reprs.add(repr(t))
            count += 1

        return count, len(data), len(reprs), len(hashes)

    def test_terminal_hashdata_depends_on_literals(self):
        reprs = set()
        hashes = set()
        def forms():
            x = SpatialCoordinate(triangle)
            i, j = indices(2)
            for d in (2, 3):
                I = Identity(d)
                for fv in (1.1, 2.2):
                    for iv in (5, 7):
                        expr = (I[0, j]*(fv*x[j]))**iv

                        reprs.add(repr(expr))
                        hashes.add(hash(expr))
                        yield compute_terminal_hashdata(expr, domain_numbering(triangle))

        c, d, r, h = self.compute_unique_terminal_hashdatas(forms())
        self.assertEqual(c, 8)
        self.assertEqual(d, c)
        self.assertEqual(r, c)
        self.assertEqual(h, c)
        self.assertEqual(len(reprs), c)
        self.assertEqual(len(hashes), c)

    def test_terminal_hashdata_depends_on_geometry(self):
        reprs = set()
        hashes = set()
        def forms():
            i, j = indices(2)
            cells = (triangle, tetrahedron, cell2D, cell3D)
            for cell in cells:

                d = cell.geometric_dimension()
                x = SpatialCoordinate(cell)
                n = FacetNormal(cell)
                r = Circumradius(cell)
                a = FacetArea(cell)
                #s = CellSurfaceArea(cell)
                v = CellVolume(cell)
                I = Identity(d)

                ws = (x, n)
                qs = (r, a, v) #, s)
                for w in ws:
                    for q in qs:
                        expr = (I[0, j]*(q*w[j]))

                        reprs.add(repr(expr))
                        hashes.add(hash(expr))
                        yield compute_terminal_hashdata(expr, domain_numbering(*cells))

        c, d, r, h = self.compute_unique_terminal_hashdatas(forms())
        self.assertEqual(c, 2*3*4) # len(ws)*len(qs)*len(cells)
        self.assertEqual(d, c)
        self.assertEqual(r, c)
        self.assertEqual(h, c)
        self.assertEqual(len(reprs), c)
        self.assertEqual(len(hashes), c)

    def test_terminal_hashdata_depends_on_form_argument_properties(self):
        reprs = set()
        hashes = set()
        nelm = 6
        nreps = 2

        # Data
        cells = (triangle, tetrahedron, cell2D, cell3D)
        degrees = (1, 2)
        families = ("CG", "Lagrange", "DG")

        def forms():
            for rep in range(nreps):
                for cell in cells:
                    d = cell.geometric_dimension()
                    for degree in degrees:
                        for family in families:
                            V = FiniteElement(family, cell, degree)
                            W = VectorElement(family, cell, degree)
                            W2 = VectorElement(family, cell, degree, dim=d+1)
                            T = TensorElement(family, cell, degree)
                            S = TensorElement(family, cell, degree, symmetry=True)
                            S2 = TensorElement(family, cell, degree, shape=(d, d), symmetry={(0, 0):(1, 1)})
                            elements = [V, W, W2, T, S, S2]
                            assert len(elements) == nelm

                            for H in elements[:nelm]:
                                # Keep number and count fixed, we're not testing that here
                                a = Argument(H, number=1)
                                c = Coefficient(H, count=1)
                                for f in (a, c):
                                    expr = inner(f, f)

                                    reprs.add(repr(expr))
                                    hashes.add(hash(expr))
                                    yield compute_terminal_hashdata(expr, domain_numbering(*cells))

        c, d, r, h = self.compute_unique_terminal_hashdatas(forms())
        c1 = nreps * len(cells) * len(degrees) * len(families) * nelm * 2 # Number of cases with repetitions
        self.assertEqual(c, c1)

        c0 = len(cells) * len(degrees) * (len(families)-1) * nelm * 2 # Number of unique cases, "CG" == "Lagrange"
        #c0 = len(cells) * len(degrees) * (len(families)) * nelm * 2 # Number of unique cases, "CG" != "Lagrange"
        self.assertEqual(d, c0)
        self.assertEqual(r, c0)
        self.assertEqual(h, c0)
        self.assertEqual(len(reprs), c0)
        self.assertEqual(len(hashes), c0)

    def test_terminal_hashdata_does_not_depend_on_coefficient_count_values_only_ordering(self):
        reprs = set()
        hashes = set()
        counts = list(range(-3, 4))
        cells = (cell1D, triangle, hexahedron)
        assert len(counts) == 7
        nreps = 1
        def forms():
            for rep in range(nreps):
                for cell in cells:
                    for k in counts:
                        V = FiniteElement("CG", cell, 2)
                        f = Coefficient(V, count=k)
                        g = Coefficient(V, count=k+2)
                        expr = inner(f, g)

                        reprs.add(repr(expr))
                        hashes.add(hash(expr))
                        yield compute_terminal_hashdata(expr, domain_numbering(*cells))

        c, d, r, h = self.compute_unique_terminal_hashdatas(forms())
        c0 = len(cells) # Number of actually unique cases from a code generation perspective
        c1 = len(counts) * c0 # Number of unique cases from a symbolic representation perspective
        self.assertEqual(len(reprs), c1)
        self.assertEqual(len(hashes), c1)
        self.assertEqual(c, nreps * c1) # number of inner loop executions in forms() above
        self.assertEqual(d, c0)
        self.assertEqual(r, c0)
        self.assertEqual(h, c0)

    def test_terminal_hashdata_does_depend_on_argument_number_values(self):
        # TODO: Include part numbers as well
        reprs = set()
        hashes = set()
        counts = list(range(4))
        cells = (cell1D, triangle, hexahedron)
        nreps = 2
        def forms():
            for rep in range(nreps):
                for cell in cells:
                    for k in counts:
                        V = FiniteElement("CG", cell, 2)
                        f = Argument(V, k)
                        g = Argument(V, k+2)
                        expr = inner(f, g)

                        reprs.add(repr(expr))
                        hashes.add(hash(expr))
                        yield compute_terminal_hashdata(expr, domain_numbering(*cells))

        c, d, r, h = self.compute_unique_terminal_hashdatas(forms())
        c0 = len(cells) * len(counts) # Number of actually unique cases from a code generation perspective
        c1 = 1 * c0 # Number of unique cases from a symbolic representation perspective
        self.assertEqual(len(reprs), c1)
        self.assertEqual(len(hashes), c1)
        self.assertEqual(c, nreps * c1) # number of inner loop executions in forms() above
        self.assertEqual(d, c0)
        self.assertEqual(r, c0)
        self.assertEqual(h, c0)

    def test_domain_signature_data_does_not_depend_on_domain_label_value(self):
        cells = [triangle, tetrahedron, hexahedron]
        s0s = set()
        s1s = set()
        s2s = set()
        for cell in cells:
            d0 = Domain(cell)
            d1 = Domain(cell, label="domain1")
            d2 = Domain(cell, label="domain2")
            s0 = d0.signature_data(domain_numbering2(*[d0]))
            s1 = d1.signature_data(domain_numbering2(*[d1]))
            s2 = d2.signature_data(domain_numbering2(*[d2]))
            self.assertEqual(s0, s1)
            self.assertEqual(s0, s2)
            s0s.add(s0)
            s1s.add(s1)
            s2s.add(s2)
        self.assertEqual(len(s0s), len(cells))
        self.assertEqual(len(s1s), len(cells))
        self.assertEqual(len(s2s), len(cells))

    def test_terminal_hashdata_does_not_depend_on_domain_label_value(self):
        reprs = set()
        hashes = set()
        labels = ["domain1", "domain2"]
        cells = [triangle, quadrilateral]
        domains = [Domain(cell, label=label) for cell in cells for label in labels]
        nreps = 2
        num_exprs = 2
        def forms():
            for rep in range(nreps):
                for domain in domains:
                    V = FiniteElement("CG", domain, 2)
                    f = Coefficient(V, count=0)
                    v = TestFunction(V)
                    x = SpatialCoordinate(domain)
                    n = FacetNormal(domain)
                    exprs = [inner(x, n), inner(f, v)]
                    assert num_exprs == len(exprs) # Assumed in checks below
                    for expr in exprs:
                        #print; print expr
                        reprs.add(repr(expr))
                        hashes.add(hash(expr))
                        # This numbering needs to be recreated to count 'domain' as 0 each time:
                        dn = domain_numbering2(*[domain])
                        yield compute_terminal_hashdata(expr, dn)

        c, d, r, h = self.compute_unique_terminal_hashdatas(forms())
        c0 = num_exprs * len(cells) # Number of actually unique cases from a code generation perspective
        c1 = num_exprs * len(domains) # Number of unique cases from a symbolic representation perspective
        self.assertEqual(len(reprs), c1)
        self.assertEqual(len(hashes), c1)
        self.assertEqual(c, nreps * c1) # number of inner loop executions in forms() above
        self.assertEqual(d, c0)
        self.assertEqual(r, c0)
        self.assertEqual(h, c0)

class MultiIndexHashDataTestCase(UflTestCase):

    def compute_unique_multiindex_hashdatas(self, hashdatas):
        count = 0
        data = set()
        hashes = set()
        reprs = set()
        for d in hashdatas:
            data.add(tuple(d))
            hashes.add(hash(tuple(d)))
            reprs.add(repr(d))
            count += 1
        return count, len(data), len(reprs), len(hashes)

    def test_multiindex_hashdata_depends_on_fixed_index_values(self):
        reprs = set()
        hashes = set()
        def hashdatas():
            for i in range(3):
                for ii in ((i,), (i, 0), (1, i)):
                    expr = MultiIndex(ii, {})
                    self.assertTrue(isinstance(expr.index_dimensions(), EmptyDictType)) # Just a side check
                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_multiindex_hashdata(expr, {})

        c, d, r, h = self.compute_unique_multiindex_hashdatas(hashdatas())
        self.assertEqual(c, 9)
        self.assertEqual(d, 9-1) # (1,0) is repeated, therefore -1
        self.assertEqual(len(reprs), 9-1)
        self.assertEqual(len(hashes), 9-1)

    def test_multiindex_hashdata_does_not_depend_on_counts(self):
        reprs = set()
        hashes = set()
        def hashdatas():
            ijs = []
            iind = indices(3)
            jind = indices(3)
            for i in iind:
                ijs.append((i,))
                for j in jind:
                    ijs.append((i, j))
                    ijs.append((j, i))
            for ij in ijs:
                expr = MultiIndex(ij, {i:2,j:3})
                reprs.add(repr(expr))
                hashes.add(hash(expr))
                yield compute_multiindex_hashdata(expr, {})
        c, d, r, h = self.compute_unique_multiindex_hashdatas(hashdatas())
        self.assertEqual(c, 3+9+9)
        self.assertEqual(d, 1+1)
        self.assertEqual(len(reprs), 3+9+9)
        self.assertEqual(len(hashes), 3+9+9)

    def test_multiindex_hashdata_depends_on_the_order_indices_are_observed(self):
        reprs = set()
        hashes = set()
        nrep = 3
        def hashdatas():
            for rep in range(nrep):
                # Resetting index_numbering for each repetition,
                # resulting in hashdata staying the same for
                # each repetition but repr and hashes changing
                # because new indices are created each repetition.
                index_numbering = {}
                i, j, k, l = indices(4)
                idims = {i:2,j:3,k:4,l:5}
                for expr in (MultiIndex((i,), idims),
                             MultiIndex((i,), idims), # r
                             MultiIndex((i, j), idims),
                             MultiIndex((j, i), idims),
                             MultiIndex((i, j), idims), # r
                             MultiIndex((i, j, k), idims),
                             MultiIndex((k, j, i), idims),
                             MultiIndex((j, i), idims)): # r
                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_multiindex_hashdata(expr, index_numbering)
        c, d, r, h = self.compute_unique_multiindex_hashdatas(hashdatas())
        self.assertEqual(c, nrep*8)
        self.assertEqual(d, 5)
        self.assertEqual(len(reprs), nrep*5)
        self.assertEqual(len(hashes), nrep*5)

    def test_multiindex_hashdata_does_not_depend_on_index_dimension(self):
        # The index dimensions are always inferred from the
        # surrounding expression, and therefore don't need
        # to be included in the form signature.
        reprs = set()
        hashes = set()
        nrep = 3
        def hashdatas():
            for rep in range(nrep):
                index_numbering = {}
                i, j = indices(2)
                idims1 = {i:1,j:2}
                idims2 = {i:3,j:4}
                for expr in (MultiIndex((i,), idims1),
                             MultiIndex((i,), idims2),
                             MultiIndex((i, j), idims1),
                             MultiIndex((i, j), idims2)):
                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_multiindex_hashdata(expr, index_numbering)
        c, d, r, h = self.compute_unique_multiindex_hashdatas(hashdatas())
        self.assertEqual(c, nrep*4)
        self.assertEqual(d, 2)
        self.assertEqual(len(reprs), nrep*4)
        self.assertEqual(len(hashes), nrep*4)


class FormSignatureTestCase(UflTestCase):

    def check_unique_signatures(self, forms):
        count = 0
        sigs = set()
        hashes = set()
        reprs = set()
        for a in forms:
            sig = compute_form_signature(a)

            sigs.add(sig)
            self.assertTrue(sig)
            hashes.add(hash(a))
            reprs.add(repr(a))
            count += 1
        self.assertEqual(len(sigs), count)
        self.assertEqual(len(reprs), count)
        self.assertEqual(len(hashes), count)

    def test_signature_is_affected_by_element_properties(self):
        def forms():
            for family in ("CG", "DG"):
                for cell in (triangle, tetrahedron, quadrilateral):
                    for degree in (1, 2):
                        V = FiniteElement(family, cell, degree)
                        u = Coefficient(V)
                        v = TestFunction(V)
                        x = SpatialCoordinate(cell)
                        w = as_vector([v]*x.shape()[0])
                        f = dot(w, u*x)
                        a = f*dx
                        yield a
        self.check_unique_signatures(forms())

    def test_signature_is_affected_by_domains(self):
        def forms():
            for cell in (cell2D, cell3D):
                for di in (1, 2):
                    for dj in (1, 2):
                        for dk in (1, 2):
                            V = FiniteElement("CG", cell, 1)
                            u = Coefficient(V)
                            a = u*dx(di) + 2*u*dx(dj) + 3*u*ds(dk)
                            yield a
        self.check_unique_signatures(forms())

    def test_signature_of_forms_with_diff(self):
        def forms():
            for cell in (cell2D, cell3D):
                for k in (1, 2, 3):
                    V = FiniteElement("CG", cell, 1)
                    W = VectorElement("CG", cell, 1)
                    u = Coefficient(V)
                    w = Coefficient(W)
                    vu = variable(u)
                    vw = variable(w)
                    f = vu*dot(vw, vu**k*vw)
                    g = diff(f, vu)
                    h = dot(diff(f, vw), FacetNormal(cell))
                    a = f*dx(1) + g*dx(2) + h*ds(0)
                    yield a
        self.check_unique_signatures(forms())

    def test_signature_of_form_depend_on_coefficient_numbering_across_integrals(self):
        cell = cell2D
        V = FiniteElement("CG", cell, 1)
        f = Coefficient(V)
        g = Coefficient(V)
        M1 = f*dx(0) + g*dx(1)
        M2 = g*dx(0) + f*dx(1)
        M3 = g*dx(0) + g*dx(1)
        self.assertTrue(compute_form_signature(M1) != compute_form_signature(M2))
        self.assertTrue(compute_form_signature(M1) != compute_form_signature(M3))
        self.assertTrue(compute_form_signature(M2) != compute_form_signature(M3))

    def test_signature_of_forms_change_with_operators(self):
        def forms():
            for cell in (cell2D, cell3D):
                V = FiniteElement("CG", cell, 1)
                u = Coefficient(V)
                v = Coefficient(V)
                fs = [(u*v)+(u/v),
                      (u+v)+(u/v),
                      (u+v)*(u/v),
                      (u*v)*(u*v),
                      (u+v)*(u*v), # (!) same
                      #(u*v)*(u+v), # (!) same
                      (u*v)+(u+v),
                      ]
                for f in fs:
                    a = f*dx
                    yield a
        self.check_unique_signatures(forms())


# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

