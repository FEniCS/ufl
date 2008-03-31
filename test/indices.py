#!/usr/bin/env python

import unittest

from ufl import *
from ufl.utilities import * 

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


# TODO: add more expressions to test as many possible combinations of index notation as feasible...


class IndexTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_vector_indices(self):
        element = VectorElement("CG", "triangle", 1)
        u = BasisFunction(element)
        f = Function(element)
        a = u[i]*f[i]*dx
        b = u[j]*f[j]*dx
    
    def test_tensor_indices(self):
        element = TensorElement("CG", "triangle", 1)
        u = BasisFunction(element)
        f = Function(element)
        a = u[i,j]*f[i,j]*dx
        b = u[j,i]*f[i,j]*dx
        c = u[j,i]*f[j,i]*dx
        try:
            d = (u[i,i]+f[j,i])*dx
            self.fail()
        except (UFLException, e):
            pass

    def test_indexed_sum1(self):
        element = VectorElement("CG", "triangle", 1)
        u = BasisFunction(element)
        f = Function(element)
        a = u[i]+f[i]
        try:
            a*dx
            self.fail()
        except (UFLException, e):
            pass

    def test_indexed_sum2(self):
        element = VectorElement("CG", "triangle", 1)
        v = BasisFunction(element)
        u = BasisFunction(element)
        f = Function(element)
        a = u[j]+f[j]+v[j]+2*v[j]+exp(u[i]*u[i])/2*f[j]
        try:
            a*dx
            self.fail()
        except (UFLException, e):
            pass

    def test_indexed_sum3(self):
        element = VectorElement("CG", "triangle", 1)
        u = BasisFunction(element)
        f = Function(element)
        try:
            a = u[i]+f[j]
            self.fail()
        except (UFLException, e):
            pass

    def test_indexed_function1(self):
        element = VectorElement("CG", "triangle", 1)
        v = BasisFunction(element)
        u = BasisFunction(element)
        f = Function(element)
        aarg = (u[i]+f[i])*v[i]
        a = exp(aarg)*dx

    def test_indexed_function2(self):
        element = VectorElement("CG", "triangle", 1)
        v = BasisFunction(element)
        u = BasisFunction(element)
        f = Function(element)
        bfun  = cos(f[0])
        left  = u[i] + f[i]
        right = v[i] * bfun
        self.assertTrue( len(left.free_indices())  == 1 and left.free_indices()[0]  == i )
        self.assertTrue( len(right.free_indices()) == 1 and right.free_indices()[0] == i )
        b = left * right * dx
    
    def test_indexed_function3(self):
        element = VectorElement("CG", "triangle", 1)
        v = BasisFunction(element)
        u = BasisFunction(element)
        f = Function(element)
        try:
            c = sin(u[i] + f[i])*dx
            self.fail()
        except (UFLException, e):
            pass
    
    def test_tensors(self):
        element = VectorElement("CG", "triangle", 1)
        v  = TestFunction(element)
        u  = TrialFunction(element)
        f  = Function(element)
        
        # legal
        vv = Vector(u[i], i)
        uu = Vector(v[j], j)
        w  = v + u
        ww = vv + uu
        self.assertTrue(vv.rank() == 1)
        self.assertTrue(uu.rank() == 1)
        self.assertTrue(w.rank() == 1)
        self.assertTrue(ww.rank() == 1)
        
        A  = Matrix(u[i]*v[j], (i,j))
        B  = Matrix(v[k]*v[k]*u[i]*v[j], (j,i))
        C  = A + A
        C  = B + B
        D  = A + B
        self.assertTrue(A.rank() == 2)
        self.assertTrue(B.rank() == 2)
        self.assertTrue(C.rank() == 2)
        self.assertTrue(D.rank() == 2)
        
        # legal
        vv = Vector([u[0], v[0]])
        ww = vv + vv
        
        A  = Matrix( u[i]*v[j], (i,j))
        B  = Matrix( (v[k]*v[k]) * u[i]*v[j], (j,i) )
        C  = A + A
        C  = B + B
        C  = A + B
        
        # legal?
        vv = Vector([u[i], v[i]])
        ww = f[i]*vv # this is well formed, ww = sum_i 
        
        # illegal?
        try:
            vv = Vector([u[i], v[j]])
            self.fail()
        except (UFLException, e):
            pass
        
        # ...


suite1 = unittest.makeSuite(IndexTestCase)

allsuites = unittest.TestSuite((suite1, ))

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=0).run(allsuites)
