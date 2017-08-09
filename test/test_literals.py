#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

__authors__ = "Martin Sandve Alnæs"
__date__ = "2011-04-14 -- 2011-04-14"

import pytest

from ufl import *
from ufl.classes import Indexed
from ufl.constantvalue import Zero, FloatValue, IntValue, ComplexValue, as_ufl


def test_zero(self):
    z1 = Zero(())
    z2 = Zero(())
    z3 = as_ufl(0)
    z4 = as_ufl(0.0)
    z5 = FloatValue(0)
    z6 = FloatValue(0.0)

    # self.assertTrue(z1 is z2)
    # self.assertTrue(z1 is z3)
    # self.assertTrue(z1 is z4)
    # self.assertTrue(z1 is z5)
    # self.assertTrue(z1 is z6)
    assert z1 == z1
    assert int(z1) == 0
    assert float(z1) == 0.0
    assert complex(z1) == 0.0 + 0.0j
    self.assertNotEqual(z1, 1.0)
    self.assertFalse(z1)

    # If zero() == 0 is to be allowed, it must not have the same hash or it will collide with 0 as key in dicts...
    self.assertNotEqual(hash(z1), hash(0.0))
    self.assertNotEqual(hash(z1), hash(0))


def test_float(self):
    f1 = as_ufl(1)
    f2 = as_ufl(1.0)
    f3 = FloatValue(1)
    f4 = FloatValue(1.0)
    f5 = 3 - FloatValue(1) - 1
    f6 = 3 * FloatValue(2) / 6

    assert f1 == f1
    self.assertNotEqual(f1, f2)  # IntValue vs FloatValue, == compares representations!
    assert f2 == f3
    assert f2 == f4
    assert f2 == f5
    assert f2 == f6


def test_int(self):
    f1 = as_ufl(1)
    f2 = as_ufl(1.0)
    f3 = IntValue(1)
    f4 = IntValue(1.0)
    f5 = 3 - IntValue(1) - 1
    f6 = 3 * IntValue(2) / 6

    assert f1 == f1
    self.assertNotEqual(f1, f2)  # IntValue vs FloatValue, == compares representations!
    assert f1 == f3
    assert f1 == f4
    assert f1 == f5
    assert f2 == f6  # Division produces a FloatValue


def test_complex(self):
    f1 = as_ufl(1 + 1j)
    f2 = as_ufl(1)
    f3 = as_ufl(1j)
    f4 = ComplexValue(1 + 1j)
    f5 = ComplexValue(1.0 + 1.0j)
    f6 = as_ufl(1.0)
    f7 = as_ufl(1.0j)

    assert f1 == f1
    assert f1 == f4 
    assert f1 == f5 # ComplexValue uses floats
    assert f1 == f2 + f3 # Type promotion of IntValue to ComplexValue with arithmetic
    assert f4 == f2 + f3
    assert f5 == f2 + f3
    assert f4 == f5
    assert f6 + f7 == f2 + f3


def test_scalar_sums(self):
    n = 10
    s = [as_ufl(i) for i in range(n)]

    for i in range(n):
        self.assertNotEqual(s[i], i+1)

    for i in range(n):
        assert s[i] == i

    for i in range(n):
        assert 0 + s[i] == i

    for i in range(n):
        assert s[i] + 0 == i

    for i in range(n):
        assert 0 + s[i] + 0 == i

    for i in range(n):
        assert 1 + s[i] - 1 == i

    assert s[1] + s[1] == 2
    assert s[1] + s[2] == 3
    assert s[1] + s[2] + s[3] == s[6]
    assert s[5] - s[2] == 3
    assert 1*s[5] == 5
    assert 2*s[5] == 10
    assert s[6]/3 == 2


def test_identity(self):
    pass  # FIXME


def test_permutation_symbol_3(self):
    e = PermutationSymbol(3)
    assert e.ufl_shape == (3, 3, 3)
    assert eval(repr(e)) == e
    for i in range(3):
        for j in range(3):
            for k in range(3):
                value = (j-i)*(k-i)*(k-j)/2
                self.assertEqual(e[i, j, k], value)
    i, j, k = indices(3)
    self.assertIsInstance(e[i, j, k], Indexed)
    x = (0, 0, 0)
    self.assertEqual((e[i, j, k] * e[i, j, k])(x), 6)


def test_permutation_symbol_n(self):
    for n in range(2, 5):  # tested with upper limit 7, but evaluation is a bit slow then
        e = PermutationSymbol(n)
        assert e.ufl_shape == (n,)*n
        assert eval(repr(e)) == e

        ii = indices(n)
        x = (0,)*n
        nfac = product(m for m in range(1, n+1))
        assert (e[ii] * e[ii])(x) == nfac


def test_unit_dyads(self):
    from ufl.tensors import unit_vectors, unit_matrices
    ei, ej = unit_vectors(2)
    self.assertEqual(as_vector((1, 0)), ei)
    self.assertEqual(as_vector((0, 1)), ej)
    eii, eij, eji, ejj = unit_matrices(2)
    self.assertEqual(as_matrix(((1, 0), (0, 0))), eii)
    self.assertEqual(as_matrix(((0, 1), (0, 0))), eij)
    self.assertEqual(as_matrix(((0, 0), (1, 0))), eji)
    self.assertEqual(as_matrix(((0, 0), (0, 1))), ejj)
