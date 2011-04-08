#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-29 -- 2008-10-29"

from ufltestcase import UflTestCase, main

from ufl import *
from ufl.constantvalue import Zero, FloatValue, IntValue, as_ufl

class BaseTestCase(UflTestCase):

    def setUp(self):
        pass

    def test_zero(self):
        z1 = Zero(())
        z2 = Zero(())
        z3 = as_ufl(0)
        z4 = as_ufl(0.0)
        z5 = FloatValue(0)
        z6 = FloatValue(0.0)
        
        #self.assertTrue(z1 is z2)
        #self.assertTrue(z1 is z3)
        #self.assertTrue(z1 is z4)
        #self.assertTrue(z1 is z5)
        #self.assertTrue(z1 is z6)
        self.assertTrue(z1 == z1)
        self.assertTrue(z1 == 0)
        self.assertTrue(z1 == 0.0)
        self.assertFalse(z1 == 1.0)
        self.assertFalse(z1)
    
    def test_float(self):
        f1 = as_ufl(1)
        f2 = as_ufl(1.0)
        f3 = FloatValue(1)
        f4 = FloatValue(1.0)
        f5 = 3 - FloatValue(1) - 1 
        f6 = 3 * FloatValue(2) / 6
        
        self.assertTrue(f1 == f1)
        self.assertTrue(f1 == f2)
        self.assertTrue(f1 == f3)
        self.assertTrue(f1 == f4)
        self.assertTrue(f1 == f5)
        self.assertTrue(f1 == f6)
    
    def test_int(self):
        f1 = as_ufl(1)
        f2 = as_ufl(1.0)
        f3 = IntValue(1)
        f4 = IntValue(1.0)
        f5 = 3 - IntValue(1) - 1 
        f6 = 3 * IntValue(2) / 6
        
        self.assertTrue(f1 == f1)
        self.assertTrue(f1 == f2)
        self.assertTrue(f1 == f3)
        self.assertTrue(f1 == f4)
        self.assertTrue(f1 == f5)
        self.assertTrue(f1 == f6)
    
    def test_scalar_sums(self):
        n = 10
        s = [as_ufl(i) for i in range(n)]
        
        for i in range(n):
            self.assertFalse(s[i] == i+1)
        
        for i in range(n):
            self.assertTrue(s[i] == i)
        
        for i in range(n):
            self.assertTrue(0 + s[i] == i)
        
        for i in range(n):
            self.assertTrue(s[i] + 0 == i)
        
        for i in range(n):
            self.assertTrue(0 + s[i] + 0 == i)
        
        for i in range(n):
            self.assertTrue(1 + s[i] - 1 == i)
        
        self.assertTrue(s[1] + s[1] == 2)
        self.assertTrue(s[1] + s[2] == 3)
        self.assertTrue(s[1] + s[2] + s[3] == s[6])
        self.assertTrue(s[5] - s[2] == 3)
        self.assertTrue(1*s[5] == 5)
        self.assertTrue(2*s[5] == 10)
        self.assertTrue(s[6]/3 == 2)

tests = [BaseTestCase]

if __name__ == "__main__":
    main()
