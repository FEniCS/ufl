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
        #a = Product()
        #a = Division()
        #a = Power()
        
        #a = Tensor()
        #a = ListVector()
        #a = ListMatrix()
        
        #a = Deviatoric()
        #a = Transposed()
        #a = Determinant()
        #a = Trace()
        #a = Cofactor()
        
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
        
        #a = PartialDerivative()
        #a = Diff()
        #a = Div()
        #a = Grad()
        #a = Curl()
        #a = Rot()
        
        #a = Restricted()
        #a = PositiveRestricted()
        #a = NegativeRestricted()

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
        l = Lhs(a)
        r = Rhs(a)
        d = Derivative(a, f1)
        b = Action(l)
        c = Action(d)
        
if __name__ == "__main__":
    unittest.main()

