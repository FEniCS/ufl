#!/usr/bin/env python

"""
Test the computation of form signatures.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

from ufl.common import EmptyDict
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
                    self.assertTrue(expr.index_dimensions() is EmptyDict) # Just a side check
                    d = compute_multiindex_hashdata(expr, {})
                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield d
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
            #sig = a.signature()
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


# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

