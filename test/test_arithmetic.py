#!/usr/bin/env python

from ufltestcase import UflTestCase, main

from ufl import *
from ufl.classes import Division, FloatValue, IntValue

class ArithmeticTestCase(UflTestCase):

    def setUp(self):
        super(ArithmeticTestCase, self).setUp()

    def tearDown(self):
        super(ArithmeticTestCase, self).tearDown()

    def test_scalar_casting(self):
        f = as_ufl(2.0)
        r = as_ufl(4)
        self.assertIsInstance(f, FloatValue)
        self.assertIsInstance(r, IntValue)
        self.assertEqual(f, 2.0)
        self.assertEqual(r, 4)

    def test_ufl_float_division(self):
        d = triangle.x[0] / 10.0 # TODO: Use mock instead of x
        self.assertIsInstance(d, Division)

    def test_float_ufl_division(self):
        d = 3.14 / triangle.x[0] # TODO: Use mock instead of x
        self.assertIsInstance(d, Division)

    def test_float_division(self):
        d = as_ufl(20.0) / 10.0
        self.assertIsInstance(d, FloatValue)
        self.assertEqual(d, 2)

    def test_int_division(self):
        # UFL treats all divisions as true division
        d = as_ufl(40) / 7
        self.assertIsInstance(d, FloatValue)
        self.assertEqual(d, 40.0 / 7.0)
        #self.assertAlmostEqual(d, 40 / 7.0, 15)

    def test_float_int_division(self):
        d = as_ufl(20.0) / 5
        self.assertIsInstance(d, FloatValue)
        self.assertEqual(d, 4)

    def test_floor_division_fails(self):
        f = as_ufl(2.0)
        r = as_ufl(4)
        s = as_ufl(5)
        self.assertRaises(NotImplementedError, lambda: r // 4)
        self.assertRaises(NotImplementedError, lambda: r // s)
        self.assertRaises(NotImplementedError, lambda: f // s)

if __name__ == "__main__":
    main()
