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


class TerminalHashDataTestCase(UflTestCase):

    def compute_unique_hashdatas(self, hashdatas):
        count = 0
        data = set()
        hashes = set()
        reprs = set()
        for d in hashdatas:
            if isinstance(d, dict):
                t = str(d.items())
            else:
                t = str(d)
            data.add(t)
            hashes.add(hash(t))
            reprs.add(repr(d))
            count += 1
        return count, len(data), len(reprs), len(hashes)

    def check_unique_hashdatas(self, hashdatas):
        c, d, r, h = self.compute_unique_hashdatas(hashdatas)
        self.assertEqual(d, c)
        self.assertEqual(r, c)
        self.assertEqual(h, c)

    def test_terminal_hashdata_depends_on_literals(self):
        reprs = set()
        hashes = set()
        def forms():
            x = triangle.x
            i, j = indices(2)
            for d in (2, 3):
                I = Identity(d)
                for fv in (1.1, 2.2):
                    for iv in (5, 7):
                        expr = (I[0,j]*(fv*x[j]))**iv

                        reprs.add(repr(expr))
                        hashes.add(hash(expr))
                        yield compute_terminal_hashdata(expr)

        c, d, r, h = self.compute_unique_hashdatas(forms())
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
            for cell in (triangle, tetrahedron, cell2D, cell3D):

                d = cell.d
                x = cell.x
                n = cell.n
                r = cell.circumradius
                a = cell.facet_area
                s = cell.surface_area
                v = cell.volume
                I = Identity(d)

                for w in (x, n):
                    for q in (r, a, s, v):
                        expr = (I[0,j]*(q*w[j]))

                        reprs.add(repr(expr))
                        hashes.add(hash(expr))
                        yield compute_terminal_hashdata(expr)

        c, d, r, h = self.compute_unique_hashdatas(forms())
        self.assertEqual(c, 4*2*4)
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
        def forms():
            for rep in range(nreps):
                for cell in (triangle, tetrahedron, cell2D, cell3D):
                    d = cell.d
                    for degree in (1, 2):
                        for family in ("CG", "Lagrange", "DG"):
                            V = FiniteElement(family, cell, degree)
                            W = VectorElement(family, cell, degree)
                            W2 = VectorElement(family, cell, degree, dim=d+1)
                            T = TensorElement(family, cell, degree)
                            S = TensorElement(family, cell, degree, symmetry=True)
                            S2 = TensorElement(family, cell, degree, shape=(d,d), symmetry={(0,0):(1,1)})
                            elements = [V, W, W2, T, S, S2]
                            assert len(elements) == nelm

                            for H in elements[:nelm]:
                                a = Argument(H, count=1)
                                c = Coefficient(H, count=1)
                                for f in (a,c):
                                    expr = inner(f,f)

                                    reprs.add(repr(expr))
                                    hashes.add(hash(expr))
                                    data = compute_terminal_hashdata(expr)
                                    yield data

        c, d, r, h = self.compute_unique_hashdatas(forms())
        c1 = nreps* 4*2*  3    *nelm*2
        c2 =        4*2* (3-1) *nelm*2
        self.assertEqual(c, c1)
        self.assertEqual(d, c2)
        self.assertEqual(r, c2)
        self.assertEqual(h, c2)
        self.assertEqual(len(reprs), c2)
        self.assertEqual(len(hashes), c2)

    def test_terminal_hashdata_does_not_depend_on_form_argument_counts(self):
        reprs = set()
        hashes = set()
        counts = list(range(-3,4))
        nreps = 2
        def forms():
            for rep in range(nreps):
                for cell in (triangle, hexahedron):
                    for k in counts:
                        V = FiniteElement("CG", cell, 2)
                        a1 = Argument(V, count=k)
                        a2 = Argument(V, count=k+2)
                        c1 = Coefficient(V, count=k)
                        c2 = Coefficient(V, count=k+2)
                        for f,g in ((a1,a2), (c1,c2)):
                            expr = inner(f,g)

                            reprs.add(repr(expr))
                            hashes.add(hash(expr))

                            data = compute_terminal_hashdata(expr)
                            keys = sorted(data.keys(), key=lambda x: x.count())
                            values = [data[k] for k in keys]
                            yield values

        c, d, r, h = self.compute_unique_hashdatas(forms())
        c1 = len(counts) * 4
        self.assertEqual(c, nreps * c1)
        self.assertEqual(d, 4)
        self.assertEqual(r, 4)
        self.assertEqual(h, 4)
        self.assertEqual(len(reprs), c1)
        self.assertEqual(len(hashes), c1)

class MultiIndexHashDataTestCase(UflTestCase):

    def compute_unique_hashdatas(self, hashdatas):
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

    def check_unique_hashdatas(self, hashdatas):
        c, d, r, h = self.compute_unique_hashdatas(hashdatas)
        self.assertEqual(d, c)
        self.assertEqual(r, c)
        self.assertEqual(h, c)

    def test_multiindex_hashdata_depends_on_fixed_index_values(self):
        reprs = set()
        hashes = set()
        def hashdatas():
            for i in range(3):
                for ii in ((i,), (i,0), (1,i)):
                    expr = MultiIndex(ii, {})
                    self.assertTrue(type(expr.index_dimensions()) is EmptyDictType) # Just a side check

                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_multiindex_hashdata(expr, {})

        c, d, r, h = self.compute_unique_hashdatas(hashdatas())
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
                    ijs.append((i,j))
                    ijs.append((j,i))
            for ij in ijs:
                expr = MultiIndex(ij, {i:2,j:3})
                d = compute_multiindex_hashdata(expr, {})
                reprs.add(repr(expr))
                hashes.add(hash(expr))
                yield d
        c, d, r, h = self.compute_unique_hashdatas(hashdatas())
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
                             MultiIndex((i,j), idims),
                             MultiIndex((j,i), idims),
                             MultiIndex((i,j), idims), # r
                             MultiIndex((i,j,k), idims),
                             MultiIndex((k,j,i), idims),
                             MultiIndex((j,i), idims)): # r
                    d = compute_multiindex_hashdata(expr, index_numbering)
                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield d
        c, d, r, h = self.compute_unique_hashdatas(hashdatas())
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
                             MultiIndex((i,j), idims1),
                             MultiIndex((i,j), idims2)):
                    d = compute_multiindex_hashdata(expr, index_numbering)
                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield d
        c, d, r, h = self.compute_unique_hashdatas(hashdatas())
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
            #sig = a.deprecated_signature()
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
                    for degree in (1,2):
                        V = FiniteElement(family, cell, degree)
                        u = Coefficient(V)
                        v = TestFunction(V)
                        x = cell.x
                        w = as_vector([v,v])
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
                    f = vu*dot(vw,vu**k*vw)
                    g = diff(f, vu)
                    h = dot(diff(f, vw), cell.n)
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
        self.assertTrue(M1.deprecated_signature() != M2.deprecated_signature())
        self.assertTrue(M1.deprecated_signature() != M3.deprecated_signature())
        self.assertTrue(M2.deprecated_signature() != M3.deprecated_signature())

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

