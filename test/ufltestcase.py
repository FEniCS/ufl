import unittest

class UflTestCase(unittest.TestCase):
    def setUp(self):
        super(UflTestCase, self).setUp()
        #print "UflTestCase.setup"

    def tearDown(self):
        #print "UflTestCase.tearDown"
        super(UflTestCase, self).tearDown()

    def assertIsInstance(self, obj, cl):
        self.assertTrue(isinstance(obj, cl))

    def assertNotInstance(self, obj, cl):
        self.assertFalse(isinstance(obj, cl))

    def assertIndices(self, expr, free_indices):
        self.assertEqual(expr.free_indices(), free_indices)

    def assertShape(self, expr, shape):
        self.assertEqual(expr.shape(), shape)

    def assertExprProperties(self, expr, shape=None, free_indices=None, terminal=None):
        if shape is not None:
            self.assertShape(expr, shape)
        if free_indices is not None:
            self.assertIndices(expr, free_indices)
        if terminal is not None:
            if terminal:
                self.assertIsInstance(expr, Terminal)
            else:
                self.assertIsInstance(expr, Operator)

def main(*args, **kwargs):
    "Hook to do something before running single file tests."
    return unittest.main(*args, **kwargs)
