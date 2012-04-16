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

    def testCoefficient(self):
        V = FiniteElement("CG", triangle, 1)
        f = Coefficient(V)
        e = f**2
        def eval_f(x):
            return x[0]*x[1]**2
        self.assertEqual(e((3,7), {f: eval_f}),
                         (3*7**2)**2)

    def testCoefficientDerivative(self):
        V = FiniteElement("CG", triangle, 1)
        f = Coefficient(V)
        e = f.dx(0)**2 + f.dx(1)**2

        def eval_f(x, derivatives):
            if not derivatives:
                return eval_f.c*x[0]*x[1]**2
            # assume only first order derivative
            d, = derivatives
            if d == 0:
                return eval_f.c*x[1]**2
            if d == 1:
                return eval_f.c*x[0]*2*x[1]
        # shows how to attach data to eval_f
        eval_f.c = 5

        self.assertEqual(e((3,7), {f: eval_f}),
                         (5*7**2)**2 + (5*3*2*7)**2)

    def test_dot(self):
        x = cell2D.x
        s = dot(x,2*x)
        e = s((5,7))
        v = 2*(5*5+7*7)
        self.assertEqual(e, v)

    def test_inner(self):
        x = cell2D.x
        xx = as_matrix(((2*x[0],3*x[0]), (2*x[1],3*x[1])))
        s = inner(xx,2*xx)
        e = s((5,7))
        v = 2*((5*2)**2 + (5*3)**2 + (7*2)**2 + (7*3)**2)
        self.assertEqual(e, v)

    def test_outer(self):
        x = cell2D.x
        xx = outer(outer(x, x), as_vector((2,3)))
        s = inner(xx,2*xx)
        e = s((5,7))
        v = 2 * (5**2 + 7**2)**2 * (2**2 + 3**2)
        self.assertEqual(e, v)

    def test_cross(self):
        x = cell3D.x
        xv = (3, 5, 7)

        # Test cross product of triplets of orthogonal
        # vectors, where |a x b| = |a| |b|
        ts = [\
              [as_vector((x[0],0,0)),
               as_vector((0,x[1],0)),
               as_vector((0,0,x[2]))],
              [as_vector((x[0],x[1],0)),
               as_vector((x[1],-x[0],0)),
               as_vector((0,0,x[2]))],
              [as_vector((0, x[0],x[1])),
               as_vector((0, x[1],-x[0])),
               as_vector((x[2], 0, 0))],
              [as_vector((x[0], 0, x[1])),
               as_vector((x[1], 0, -x[0])),
               as_vector((0, x[2], 0))],
              ]
        for t in ts:
            for i in range(3):
                for j in range(3):
                    cij = cross(t[i],t[j])
                    dij = dot(cij,cij)
                    eij = dij(xv)
                    tni = dot(t[i],t[i])(xv)
                    tnj = dot(t[j],t[j])(xv)
                    vij = tni*tnj if i != j else 0
                    self.assertEqual(eij, vij)

    def xtest_dev(self):
        x = cell2D.x
        xv = (5, 7)
        xx = outer(x, x)
        s1 = dev(2*xx)
        s2 = 2*(xx - xx.T) # FIXME
        e = inner(s1,s1)(xv)
        v = inner(s2,s2)(xv)
        self.assertEqual(e, v)

    def test_skew(self):
        x = cell2D.x
        xv = (5, 7)
        xx = outer(x, x)
        s1 = skew(2*xx)
        s2 = (xx - xx.T)
        e = inner(s1,s1)(xv)
        v = inner(s2,s2)(xv)
        self.assertEqual(e, v)

    def test_sym(self):
        x = cell2D.x
        xv = (5, 7)
        xx = outer(x, x)
        s1 = sym(2*xx)
        s2 = (xx + xx.T)
        e = inner(s1,s1)(xv)
        v = inner(s2,s2)(xv)
        self.assertEqual(e, v)

    def test_tr(self):
        x = cell2D.x
        xv = (5, 7)
        xx = outer(x, x)
        s = tr(2*xx)
        e = s(xv)
        v = 2*sum(xv[i]**2 for i in (0,1))
        self.assertEqual(e, v)

    def test_det2D(self):
        x = cell2D.x
        xv = (5, 7)
        a, b = 6.5, -4
        xx = as_matrix(((x[0],x[1]),(a,b)))
        s = det(2*xx)
        e = s(xv)
        v = 2**2 * (5*b - 7*a)
        self.assertEqual(e, v)

    def xtest_det3D(self): # FIXME
        x = cell3D.x
        xv = (5, 7, 9)
        a, b, c = 6.5, -4, 3
        d, e, f = 2, 3, 4
        xx = as_matrix(((x[0],x[1], x[2]),
                        (a,b,c),
                        (d,e,f)))
        s = det(2*xx)
        e = s(xv)
        v = 2**3 * (xv[0]*(b*f - e*c) - xv[1]*(a*f - c*d) + xv[2]*(a*e - b*d))
        self.assertEqual(e, v)

    def test_cofac(self):
        pass # TODO

    def test_inv(self):
        pass # TODO

if __name__ == "__main__":
    main()
