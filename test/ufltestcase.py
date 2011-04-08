import unittest

class UflTestCase(unittest.TestCase):
    def setUp(self):
        super(UflTestCase, self).setUp()
        print "UflTestCase.setup"

    def tearDown(self):
        print "UflTestCase.tearDown"
        super(UflTestCase, self).tearDown()

    def assertIsInstance(self, obj, cl):
        self.assertTrue(isinstance(obj, cl))
