#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-13 -- 2009-02-13"

from ufltestcase import UflTestCase, main
import math
        
from ufl import *
from ufl.constantvalue import as_ufl

class EvaluateTestCase(UflTestCase):

    def testScalars(self):
        s = as_ufl(123)
        e = s((5,7))
        v = 123
        self.assertEqual(e, v)

    def testZero(self):
        s = as_ufl(0)
        e = s((5,7))
        v = 0
        self.assertEqual(e, v)

    def testIdentity(self):
        cell = triangle
        I = Identity(cell.d)
        
        s = 123*I[0,0]
        e = s((5,7))
        v = 123
        self.assertEqual(e, v)
        
        s = 123*I[1,0]
        e = s((5,7))
        v = 0
        self.assertEqual(e, v)

    def testCoords(self):
        cell = triangle
        x = cell.x
        s = x[0] + x[1]
        e = s((5,7))
        v = 5 + 7
        self.assertEqual(e, v)

    def testFunction1(self):
        cell = triangle
        element = FiniteElement("CG", cell, 1)
        f = Coefficient(element)
        s = 3*f
        e = s((5,7), { f: 123 })
        v = 3*123
        self.assertEqual(e, v)

    def testFunction2(self):
        cell = triangle
        element = FiniteElement("CG", cell, 1)
        f = Coefficient(element)
        def g(x):
            return x[0]
        s = 3*f
        e = s((5,7), { f: g })
        v = 3*5
        self.assertEqual(e, v)

    def testArgument2(self):
        cell = triangle
        element = FiniteElement("CG", cell, 1)
        f = Argument(element)
        def g(x):
            return x[0]
        s = 3*f
        e = s((5,7), { f: g })
        v = 3*5
        self.assertEqual(e, v)

    def testAlgebra(self):
        cell = triangle
        x = cell.x
        s = 3*(x[0] + x[1]) - 7 + x[0]**(x[1]/2)
        e = s((5,7))
        v = 3*(5. + 7.) - 7 + 5.**(7./2)
        self.assertEqual(e, v)

    def testIndexSum(self):
        cell = triangle
        x = cell.x
        i, = indices(1)
        s = x[i]*x[i]
        e = s((5,7))
        v = 5**2 + 7**2
        self.assertEqual(e, v)
    
    def testIndexSum2(self):
        cell = triangle
        x = cell.x
        I = Identity(cell.d)
        i, j = indices(2)
        s = (x[i]*x[j])*I[i,j]
        e = s((5,7))
        #v = sum_i sum_j x_i x_j delta_ij = x_0 x_0 + x_1 x_1
        v = 5**2 + 7**2
        self.assertEqual(e, v)

    def testMathFunctions(self):
        x = triangle.x[0]
        
        s = sin(x)
        e = s((5,7))
        v = math.sin(5)
        self.assertEqual(e, v)
        
        s = cos(x)
        e = s((5,7))
        v = math.cos(5)
        self.assertEqual(e, v)

        s = tan(x)
        e = s((5,7))
        v = math.tan(5)
        self.assertEqual(e, v)
        
        s = ln(x)
        e = s((5,7))
        v = math.log(5)
        self.assertEqual(e, v)
        
        s = exp(x)
        e = s((5,7))
        v = math.exp(5)
        self.assertEqual(e, v)
        
        s = sqrt(x)
        e = s((5,7))
        v = math.sqrt(5)
        self.assertEqual(e, v)

    def testListTensor(self):
        x, y = triangle.x[0], triangle.x[1]
        
        m = as_matrix([[x, y], [-y, -x]])
        
        s = m[0,0] + m[1,0] + m[0,1] + m[1,1]
        e = s((5,7))
        v = 0
        self.assertEqual(e, v)
        
        s = m[0,0] * m[1,0] * m[0,1] * m[1,1]
        e = s((5,7))
        v = 5**2*7**2
        self.assertEqual(e, v)

    def testComponentTensor1(self):
        x = triangle.x
        m = as_vector(x[i], i)
        
        s = m[0] * m[1]
        e = s((5,7))
        v = 5*7
        self.assertEqual(e, v)

    def testComponentTensor2(self):
        x = triangle.x
        xx = outer(x,x)
        
        m = as_matrix(xx[i,j], (i,j))
        
        s = m[0,0] + m[1,0] + m[0,1] + m[1,1]
        e = s((5,7))
        v = 5*5 + 5*7 + 5*7 + 7*7
        self.assertEqual(e, v)

    def testComponentTensor3(self):
        x = triangle.x
        xx = outer(x,x)
        
        m = as_matrix(xx[i,j], (i,j))
        
        s = m[0,0] * m[1,0] * m[0,1] * m[1,1]
        e = s((5,7))
        v = 5*5 * 5*7 * 5*7 * 7*7
        self.assertEqual(e, v)


if __name__ == "__main__":
    main()
