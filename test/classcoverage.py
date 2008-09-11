#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-06 -- 2008-09-06"

import unittest

from ufl import *
from ufl.classes import * 
from ufl.algorithms import * 

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


class ClasscoverageTest(unittest.TestCase):

    def setUp(self):
        pass

    def testAll(self):
        
        # --- Elements:
        
        e0 = FiniteElement("CG", "triangle", 1)
        e1 = VectorElement("CG", "triangle", 1)
        e2 = TensorElement("CG", "triangle", 1)
        e3 = MixedElement(e0, e1, e2)
        
        # --- Terminals:
        
        v0 = BasisFunction(e0)
        v1 = BasisFunction(e1)
        v2 = BasisFunction(e2)
        v3 = BasisFunction(e3)
        
        f0 = Function(e0)
        f1 = Function(e1)
        f2 = Function(e2)
        f3 = Function(e3)
        
        c = Constant("triangle")
        
        a = FloatValue(1.23)
        I = Identity(2)
        
        n = FacetNormal()
        
        a = Variable(v0)
        a = Variable(v1)
        a = Variable(v2)
        a = Variable(v3)
        a = Variable(f0)
        a = Variable(f1)
        a = Variable(f2)
        a = Variable(f3)
        
        #a = MultiIndex()
        
        # --- Non-terminals:
        
        #a = Indexed()
        a = v2[i,j]
        a = v2[0,k]
        a = v2[l,1]
        a = f2[i,j]
        a = f2[0,k]
        a = f2[l,1]
        
        a = Inverse(I)
        a = Inverse(v2)
        a = Inverse(f2)
        
        for v in (v0,v1,v2,v3):
            for f in (f0,f1,f2,f3):
                a = Outer(v, f)
        
        for v,f in zip((v0,v1,v2,v3), (f0,f1,f2,f3)):
            a = Inner(v, f)
        
        for v in (v1,v2,v3):
            for f in (f1,f2,f3):
                a = Dot(v, f)
        
        a = Cross(v1, f1)
        
        #a = Sum()
        a = v0 + f0 + v0
        a = v1 + f1 + v1
        a = v2 + f2 + v2
        #a = Product()
        a = 3*v0*(2.0*v0)*f0*(v0*3.0)
        #a = Division()
        a = v0 / 2.0
        a = v0 / f0
        a = v0 / (f0 + 7)
        #a = Power()
        a = f0**3
        a = (f0*2)**1.23
        
        a = Tensor(v1[i]*f1[j], (i,j))
        #a = ListVector()
        a = Vector([1.0, 2.0*f0, f0**2])
        self.assertTrue(a.shape() == (3,))
        #a = ListMatrix()
        a = Matrix([[1.0, 2.0*f0, f0**2],
                    [1.0, 2.0*f0, f0**2]])
        self.assertTrue(a.shape() == (2,3))
        
        # TODO:
        #a = Deviatoric()
        #a = Transposed()
        #a = Determinant()
        #a = Trace()
        #a = Cofactor()
        
        # TODO:
        #a = LE()
        #a = EQ()
        #a = NE()
        #a = LT()
        #a = GE()
        #a = GT()
        #a = Conditional()
        
        a = Abs(f0)
        a = Mod(f0, 3.0)
        a = Mod(3.0, f0)
        a = Sqrt(f0)
        a = Cos(f0)
        a = Sin(f0)
        a = Exp(f0)
        a = Ln(f0)
        
        a = Abs(1.0)
        a = Sqrt(1.0)
        a = Cos(1.0)
        a = Sin(1.0)
        a = Exp(1.0)
        a = Ln(1.0)
        
        # TODO:
        #a = PartialDerivative()
        #a = Diff()
        
        a = Div(v1)
        a = Div(f1)
        a = Div(v2)
        a = Div(f2)
        a = Div(Outer(f1,f1))
        
        a = Grad(v0)
        a = Grad(f0)
        a = Grad(v1)
        a = Grad(f1)
        a = Grad(f0*v0)
        a = Grad(f0*v1)

        a = Curl(v1)
        a = Curl(f1)
        a = Rot(v1)
        a = Rot(f1)
        
        a = PositiveRestricted(v0)
        a = v0('+')
        a = v0('+')*f0
        
        a = NegativeRestricted(v0)
        a = v0('-')
        a = v0('-') + f0

        # --- Integrals:

        a = v0*dx
        a = v0*dx(0)
        a = v0*dx(1)
        a = v0*ds
        a = v0*ds(0)
        a = v0*ds(1)
        a = v0*dS
        a = v0*dS(0)
        a = v0*dS(1)
        
        a = v0*dot(v1,f1)*dx
        a = v0*dot(v1,f1)*dx(0)
        a = v0*dot(v1,f1)*dx(1)
        a = v0*dot(v1,f1)*ds
        a = v0*dot(v1,f1)*ds(0)
        a = v0*dot(v1,f1)*ds(1)
        a = v0*dot(v1,f1)*dS
        a = v0*dot(v1,f1)*dS(0)
        a = v0*dot(v1,f1)*dS(1)
        
        # --- Form transformations:
        
        a = f0*v0*dx + f0*v0*dot(f1,v1)*dx
        b = Lhs(a)
        c = Rhs(a)
        d = Derivative(a, f1)
        e = Action(b)
        f = Action(d)
        
if __name__ == "__main__":
    unittest.main()

