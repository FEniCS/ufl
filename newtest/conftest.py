
import pytest

class Tester:

    def assertTrue(self, a):
        assert a

    def assertFalse(self, a):
        assert not a

    def assertEqual(self, a, b):
        assert a == b

    def assertNotEqual(self, a, b):
        assert a != b

    def assertIsInstance(self, obj, cls):
        assert isinstance(obj, cls)

    def assertRaises(self, e, f):
        assert pytest.raises(e, f)

    def assertEqualTotalShape(self, value, expected):
        self.assertEqual(value.ufl_shape, expected.ufl_shape)
        self.assertEqual(set(value.free_indices()), set(expected.free_indices()))
        self.assertEqual(value.index_dimensions(), expected.index_dimensions())

@pytest.fixture(scope="session")
def self():
    return Tester()
