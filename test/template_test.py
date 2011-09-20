#!/usr/bin/env python

"""
This is a template file you can copy when making a new test case.
Begin by copying this file to a filename matching test_*.py.
The tests in the file will then automatically be run by ./test.py.
Next look at the TODO markers below for places to edit.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

# TODO: Import only what you need from classes and algorithms:
#from ufl.classes import ...
#from ufl.algorithms import ...

# TODO: Rename test case class to something unique and descriptive:
class TemplateTestCase(UflTestCase):

    def setUp(self):
        super(TemplateTestCase, self).setUp()
        # TODO: If needed, add shared code here for setting up a test fixture

    def tearDown(self):
        # TODO: If needed, add shared code here for tearing down a test fixture
        super(TemplateTestCase, self).tearDown()

    # TODO: Add as many test_foo() functions as needed, use descriptive names
    def test_doing_this_should_have_that_effect(self):
        # TODO: Use the most descriptive assertion function, e.g.:
        self.assertTrue(42)
        self.assertEqual("hi", "h" + "i")
        self.assertIsInstance(triangle, Cell)
        self.assertRaises(NameError, lambda: unknown)

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

