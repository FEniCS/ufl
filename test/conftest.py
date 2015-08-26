# -*- coding: utf-8 -*-

import pytest

import ufl
from ufl import as_ufl, inner, dx
from ufl.algorithms import compute_form_data

class Tester:

    def assertTrue(self, a):
        assert a

    def assertFalse(self, a):
        assert not a

    def assertEqual(self, a, b):
        assert a == b

    def assertAlmostEqual(self, a, b):
        assert abs(a-b) < 1e-7

    def assertNotEqual(self, a, b):
        assert a != b

    def assertIsInstance(self, obj, cls):
        assert isinstance(obj, cls)

    def assertNotIsInstance(self, obj, cls):
        assert not isinstance(obj, cls)

    def assertRaises(self, e, f):
        assert pytest.raises(e, f)

    def assertEqualTotalShape(self, value, expected):
        self.assertEqual(value.ufl_shape, expected.ufl_shape)
        self.assertEqual(value.ufl_free_indices, expected.ufl_free_indices)
        self.assertEqual(value.ufl_index_dimensions, expected.ufl_index_dimensions)

    def assertSameIndices(self, expr, free_indices):
        self.assertEqual(expr.ufl_free_indices, tuple(sorted(i.count() for i in free_indices)))

    def assertEqualAfterPreprocessing(self, a, b):
        a2 = compute_form_data(a*dx).preprocessed_form
        b2 = compute_form_data(b*dx).preprocessed_form
        self.assertEqual(a2, b2)

    def assertEqualValues(self, A, B):
        B = as_ufl(B)
        self.assertEqual(A.ufl_shape, B.ufl_shape)
        self.assertEqual(inner(A-B, A-B)(None), 0)

@pytest.fixture(scope="session")
def self():
    return Tester()

_all_cells = [ufl.interval, ufl.triangle, ufl.tetrahedron]
@pytest.fixture(params=_all_cells)
def cell(request):
    return request.param
