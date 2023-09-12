#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
"""
Test the computation of form signatures.
"""

from itertools import chain

import pytest

from ufl import *
from ufl.algorithms.signature import (compute_multiindex_hashdata,
                                      compute_terminal_hashdata)
from ufl.classes import FixedIndex, MultiIndex
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

# TODO: Test compute_terminal_hashdata
#   TODO: Check that form argument counts only affect the sig by their relative ordering
#   TODO: Check that all other relevant terminal propeties affect the terminal_hashdata

# TODO: Test that operator types affect the sig
# TODO: Test that we do not get collisions for some large sets of generated forms
# TODO: How do we know that we have tested the signature reliably enough?


def domain_numbering(*cells):
    renumbering = {}
    for i, cell in enumerate(cells):
        domain = as_domain(cell)
        renumbering[domain] = i
    return renumbering


def test_domain_signatures_of_cell2domains(self):
    all_cells = (interval, quadrilateral, hexahedron, triangle, tetrahedron)
    for cell in all_cells:
        # Equality holds when constructing two domains from a cell:
        assert as_domain(cell) == as_domain(cell)
        # Hash value holds when constructing two domains from a cell:
        assert hash(as_domain(cell)) == hash(as_domain(cell))
        # Signature data holds when constructing two domains from a cell:
        D1 = as_domain(cell)
        D2 = as_domain(cell)
        self.assertEqual(D1._ufl_signature_data_({D1: 0}),
                         D2._ufl_signature_data_({D2: 0}))


def compute_unique_terminal_hashdatas(hashdatas):
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
        # print t

        # Add the hashdata values tuple to sets based on itself, its hash,
        # and its repr (not sure why I included repr anymore?)
        hashes.add(hash(t))  # This will fail if t is not hashable, which it should be!
        data.add(t)
        reprs.add(repr(t))
        count += 1

    return count, len(data), len(reprs), len(hashes)


def test_terminal_hashdata_depends_on_literals(self):
    reprs = set()
    hashes = set()

    def forms():
        i, j = indices(2)
        for d in (2, 3):
            domain = as_domain({2: triangle, 3: tetrahedron}[d])
            x = SpatialCoordinate(domain)
            I = Identity(d)
            for fv in (1.1, 2.2):
                for iv in (5, 7):
                    expr = (I[0, j]*(fv*x[j]))**iv

                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_terminal_hashdata(expr, {domain: 0})

    c, d, r, h = compute_unique_terminal_hashdatas(forms())
    assert c == 8
    assert d == c
    assert r == c
    assert h == c
    assert len(reprs) == c
    assert len(hashes) == c


def test_terminal_hashdata_depends_on_geometry(self):
    reprs = set()
    hashes = set()

    def forms():
        i, j = indices(2)
        cells = (triangle, tetrahedron)
        for cell in cells:

            d = cell.geometric_dimension()
            x = SpatialCoordinate(cell)
            n = FacetNormal(cell)
            h = CellDiameter(cell)
            r = Circumradius(cell)
            a = FacetArea(cell)
            # s = CellSurfaceArea(cell)
            v = CellVolume(cell)
            I = Identity(d)

            ws = (x, n)
            qs = (h, r, a, v)  # , s)
            for w in ws:
                for q in qs:
                    expr = (I[0, j]*(q*w[j]))

                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_terminal_hashdata(expr, domain_numbering(*cells))

    c, d, r, h = compute_unique_terminal_hashdatas(forms())
    assert c == 2*4*2  # len(ws)*len(qs)*len(cells)
    assert d == c
    assert r == c
    assert h == c
    assert len(reprs) == c
    assert len(hashes) == c


def test_terminal_hashdata_depends_on_form_argument_properties(self):
    reprs = set()
    hashes = set()
    nelm = 5
    nreps = 2

    # Data
    cells = (triangle, tetrahedron)
    degrees = (1, 2)
    families = (("Lagrange", H1), ("Lagrange", H1), ("Discontinuous Lagrange", L2))

    def forms():
        for rep in range(nreps):
            for cell in cells:
                d = cell.geometric_dimension()
                for degree in degrees:
                    for family, sobolev in families:
                        V = FiniteElement(family, cell, degree, (), (), "identity", sobolev)
                        W = FiniteElement(family, cell, degree, (d, ), (d, ), "identity", sobolev)
                        W2 = FiniteElement(family, cell, degree, (d+1, ), (d+1, ), "identity", sobolev)
                        T = FiniteElement(family, cell, degree, (d, d), (d, d), "identity", sobolev)
                        if d == 2:
                            S = FiniteElement(family, cell, degree, (2, 2), (3, ), "identity", sobolev, component_map={
                                (0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2})
                        else:
                            assert d == 3
                            S = FiniteElement(family, cell, degree, (3, 3), (6, ), "identity", sobolev, component_map={
                                (0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 1, (1, 1): 3, (1, 2): 4, (2, 0): 2, (2, 1): 4, (2, 2): 5})
                        elements = [V, W, W2, T, S]
                        assert len(elements) == nelm

                        for H in elements[:nelm]:
                            # Keep number and count fixed, we're not testing that here
                            a = Argument(H, number=1)
                            c = Coefficient(H, count=1)
                            renumbering = domain_numbering(*cells)
                            renumbering[c] = 0
                            for f in (a, c):
                                expr = inner(f, f)

                                reprs.add(repr(expr))
                                hashes.add(hash(expr))
                                yield compute_terminal_hashdata(expr, renumbering)

    c, d, r, h = compute_unique_terminal_hashdatas(forms())
    c1 = nreps * len(cells) * len(degrees) * len(families) * nelm * 2
    assert c == c1

    c0 = len(cells) * len(degrees) * (len(families)-1) * nelm * 2
    assert d == c0
    assert r == c0
    assert h == c0
    assert len(reprs) == c0
    assert len(hashes) == c0


def test_terminal_hashdata_does_not_depend_on_coefficient_count_values_only_ordering(self):
    reprs = set()
    hashes = set()
    counts = list(range(-3, 4))
    cells = (interval, triangle, hexahedron)
    assert len(counts) == 7
    nreps = 1

    def forms():
        for rep in range(nreps):
            for cell in cells:
                for k in counts:
                    V = FiniteElement("Lagrange", cell, 2, (), (), "identity", H1)
                    f = Coefficient(V, count=k)
                    g = Coefficient(V, count=k+2)
                    expr = inner(f, g)

                    renumbering = domain_numbering(*cells)
                    renumbering[f] = 0
                    renumbering[g] = 1

                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_terminal_hashdata(expr, renumbering)

    c, d, r, h = compute_unique_terminal_hashdatas(forms())
    c0 = len(cells)  # Number of actually unique cases from a code generation perspective
    c1 = len(counts) * c0  # Number of unique cases from a symbolic representation perspective
    assert len(reprs) == c1
    assert len(hashes) == c1
    assert c == nreps * c1  # number of inner loop executions in forms() above
    assert d == c0
    assert r == c0
    assert h == c0


def test_terminal_hashdata_does_depend_on_argument_number_values(self):
    # TODO: Include part numbers as well
    reprs = set()
    hashes = set()
    counts = list(range(4))
    cells = (interval, triangle, hexahedron)
    nreps = 2

    def forms():
        for rep in range(nreps):
            for cell in cells:
                for k in counts:
                    V = FiniteElement("Lagrange", cell, 2, (), (), "identity", H1)
                    f = Argument(V, k)
                    g = Argument(V, k+2)
                    expr = inner(f, g)

                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_terminal_hashdata(expr, domain_numbering(*cells))

    c, d, r, h = compute_unique_terminal_hashdatas(forms())
    c0 = len(cells) * len(counts)  # Number of actually unique cases from a code generation perspective
    c1 = 1 * c0  # Number of unique cases from a symbolic representation perspective
    assert len(reprs) == c1
    assert len(hashes) == c1
    self.assertEqual(c, nreps * c1)  # number of inner loop executions in forms() above
    assert d == c0
    assert r == c0
    assert h == c0


def test_domain_signature_data_does_not_depend_on_domain_label_value(self):
    cells = [triangle, tetrahedron, hexahedron]
    s0s = set()
    s1s = set()
    s2s = set()
    for cell in cells:
        d0 = Mesh(cell)
        d1 = Mesh(cell, ufl_id=1)
        d2 = Mesh(cell, ufl_id=2)
        s0 = d0._ufl_signature_data_({d0: 0})
        s1 = d1._ufl_signature_data_({d1: 0})
        s2 = d2._ufl_signature_data_({d2: 0})
        assert s0 == s1
        assert s0 == s2
        s0s.add(s0)
        s1s.add(s1)
        s2s.add(s2)
    assert len(s0s) == len(cells)
    assert len(s1s) == len(cells)
    assert len(s2s) == len(cells)


def test_terminal_hashdata_does_not_depend_on_domain_label_value(self):
    reprs = set()
    hashes = set()
    ufl_ids = [1, 2]
    cells = [triangle, quadrilateral]
    domains = [Mesh(cell, ufl_id=ufl_id) for cell in cells for ufl_id in ufl_ids]
    nreps = 2
    num_exprs = 2

    def forms():
        for rep in range(nreps):
            for domain in domains:
                V = FunctionSpace(domain, FiniteElement("Lagrange", domain.ufl_cell(), 2, (), (), "identity", H1))
                f = Coefficient(V, count=0)
                v = TestFunction(V)
                x = SpatialCoordinate(domain)
                n = FacetNormal(domain)
                exprs = [inner(x, n), inner(f, v)]
                assert num_exprs == len(exprs)  # Assumed in checks below

                # This numbering needs to be recreated to count 'domain' and 'f' as 0 each time:
                renumbering = {f: 0, domain: 0}

                for expr in exprs:
                    reprs.add(repr(expr))
                    hashes.add(hash(expr))
                    yield compute_terminal_hashdata(expr, renumbering)

    c, d, r, h = compute_unique_terminal_hashdatas(forms())
    c0 = num_exprs * len(cells)  # Number of actually unique cases from a code generation perspective
    c1 = num_exprs * len(domains)  # Number of unique cases from a symbolic representation perspective
    assert len(reprs) == c1
    assert len(hashes) == c1
    self.assertEqual(c, nreps * c1)  # number of inner loop executions in forms() above
    assert d == c0
    assert r == c0
    assert h == c0


def compute_unique_multiindex_hashdatas(hashdatas):
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
                jj = tuple(FixedIndex(j) for j in ii)
                expr = MultiIndex(jj)
                reprs.add(repr(expr))
                hashes.add(hash(expr))
                yield compute_multiindex_hashdata(expr, {})

    c, d, r, h = compute_unique_multiindex_hashdatas(hashdatas())
    assert c == 9
    assert d == 9-1  # (1,0 is repeated, therefore -1)
    assert len(reprs) == 9-1
    assert len(hashes) == 9-1


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
            expr = MultiIndex(ij)
            reprs.add(repr(expr))
            hashes.add(hash(expr))
            yield compute_multiindex_hashdata(expr, {})
    c, d, r, h = compute_unique_multiindex_hashdatas(hashdatas())
    assert c == 3+9+9
    assert d == 1+1
    assert len(reprs) == 3+9+9
    assert len(hashes) == 3+9+9


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
            for expr in (MultiIndex((i,)),
                         MultiIndex((i,)),  # r
                         MultiIndex((i, j)),
                         MultiIndex((j, i)),
                         MultiIndex((i, j)),  # r
                         MultiIndex((i, j, k)),
                         MultiIndex((k, j, i)),
                         MultiIndex((j, i))):  # r
                reprs.add(repr(expr))
                hashes.add(hash(expr))
                yield compute_multiindex_hashdata(expr, index_numbering)
    c, d, r, h = compute_unique_multiindex_hashdatas(hashdatas())
    assert c == nrep*8
    assert d == 5
    assert len(reprs) == nrep*5
    assert len(hashes) == nrep*5


def check_unique_signatures(forms):
    count = 0
    sigs = set()
    sigs2 = set()
    hashes = set()
    reprs = set()
    for a in forms:
        sig = a.signature()
        sig2 = a.signature()
        sigs.add(sig)
        sigs2.add(sig2)
        assert sig
        hashes.add(hash(a))
        reprs.add(repr(a))
        count += 1
    assert len(sigs) == count
    assert len(sigs2) == count
    assert len(reprs) == count
    assert len(hashes) == count


def test_signature_is_affected_by_element_properties(self):
    def forms():
        for family, sobolev in (("Lagrange", H1), ("Discontinuous Lagrange", L2)):
            for cell in (triangle, tetrahedron, quadrilateral):
                for degree in (1, 2):
                    V = FiniteElement(family, cell, degree, (), (), "identity", sobolev)
                    u = Coefficient(V)
                    v = TestFunction(V)
                    x = SpatialCoordinate(cell)
                    w = as_vector([v]*x.ufl_shape[0])
                    f = dot(w, u*x)
                    a = f*dx
                    yield a
    check_unique_signatures(forms())


def test_signature_is_affected_by_domains(self):
    def forms():
        for cell in (triangle, tetrahedron):
            for di in (1, 2):
                for dj in (1, 2):
                    for dk in (1, 2):
                        V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
                        u = Coefficient(V)
                        a = u*dx(di) + 2*u*dx(dj) + 3*u*ds(dk)
                        yield a
    check_unique_signatures(forms())


def test_signature_of_forms_with_diff(self):
    def forms():
        for cell in (triangle, tetrahedron):
            for k in (1, 2, 3):
                d = cell.geometric_dimension()
                V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
                W = FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1)
                u = Coefficient(V)
                w = Coefficient(W)
                vu = variable(u)
                vw = variable(w)
                f = vu*dot(vw, vu**k*vw)
                g = diff(f, vu)
                h = dot(diff(f, vw), FacetNormal(cell))
                a = f*dx(1) + g*dx(2) + h*ds(0)
                yield a
    check_unique_signatures(forms())


def test_signature_of_form_depend_on_coefficient_numbering_across_integrals(self):
    cell = triangle
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    f = Coefficient(V)
    g = Coefficient(V)
    M1 = f*dx(0) + g*dx(1)
    M2 = g*dx(0) + f*dx(1)
    M3 = g*dx(0) + g*dx(1)
    self.assertTrue(M1.signature() != M2.signature())
    self.assertTrue(M1.signature() != M3.signature())
    self.assertTrue(M2.signature() != M3.signature())


def test_signature_of_forms_change_with_operators(self):
    def forms():
        for cell in (triangle, tetrahedron):
            V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
            u = Coefficient(V)
            v = Coefficient(V)
            fs = [(u*v)+(u/v),
                  (u+v)+(u/v),
                  (u+v)*(u/v),
                  (u*v)*(u*v),
                  (u+v)*(u*v),  # H1 same
                  # (u*v)*(u+v), # H1 same
                  (u*v)+(u+v),
                  ]
            for f in fs:
                a = f*dx
                yield a
    check_unique_signatures(forms())
