
import pytest

class Tester:

    def assertEqual(self, a, b):
        assert a == b

    def assertNotEqual(self, a, b):
        assert a != b

    def assertIsInstance(self, obj, cls):
        assert isinstance(obj, cls)

    def assertRaises(self, e, f):
        assert pytest.raises(e, f)

@pytest.fixture(scope="session")
def self():
    return Tester()
