#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-06 -- 2008-09-17"

import unittest

from ufl import *
from ufl.classes import * 
from ufl.algorithms import * 
from ufl.indexing import DefaultDim

# disable log output
import logging
logging.basicConfig(level=logging.CRITICAL)


def test_object(a, shape, free_indices):
    # Test reproduction via repr string
    r = repr(a)
    e = eval(r, globals())
    assert hash(a) == hash(e)
    
    # Can't really test str more than that it exists
    s = str(a)
    
    # Check that some properties are at least available
    fi = a.free_indices()
    sh = a.shape()
    
    # Compare with provided properties
    if free_indices is not None:
        assert len(set(fi) ^ set(free_indices)) == 0
    if shape is not None:
        if sh != shape:
            print "sh:", sh
            print "shape:", shape
        assert sh == shape # FIXME: Better comparison.

def test_form(a):
    # Test reproduction via repr string
    r = repr(a)
    e = eval(r, globals())
    assert hash(a) == hash(e)
    
    # Can't really test str more than that it exists
    s = str(a)

class ClasscoverageTest(unittest.TestCase):

    def setUp(self):
        pass

    def testAll(self):
        
        # --- Elements:
        polygon, dim = "triangle", 2
        
        e0 = FiniteElement("CG", polygon, 1)
        e1 = VectorElement("CG", polygon, 1)
        e2 = TensorElement("CG", polygon, 1)
        e3 = MixedElement(e0, e1, e2)
        
        # --- Terminals:
        
        v0 = BasisFunction(e0)
        v1 = BasisFunction(e1)
        v2 = BasisFunction(e2)
        v3 = BasisFunction(e3)
        
        test_object(v0, (), ())
        test_object(v1, (dim,), ())
        test_object(v2, (dim,dim), ())
        test_object(v3, (dim*dim+dim+1,), ())
        
        f0 = Function(e0)
        f1 = Function(e1)
        f2 = Function(e2)
        f3 = Function(e3)
        
        test_object(f0, (), ())
        test_object(f1, (dim,), ())
        test_object(f2, (dim,dim), ())
        test_object(f3, (dim*dim+dim+1,), ())
        
        c = Constant(polygon)
        test_object(c, (), ())
        
        a = FloatValue(1.23)
        test_object(a, (), ())
        
        I = Identity(2)
        test_object(I, (dim,dim), ())
        
        n = FacetNormal()
        test_object(n, (DefaultDim,), ())
        
        a = Variable(v0)
        test_object(a, (), ())
        a = Variable(v1)
        test_object(a, (dim,), ())
        a = Variable(v2)
        test_object(a, (dim,dim), ())
        a = Variable(v3)
        test_object(a, (dim*dim+dim+1,), ())
        a = Variable(f0)
        test_object(a, (), ())
        a = Variable(f1)
        test_object(a, (dim,), ())
        a = Variable(f2)
        test_object(a, (dim,dim), ())
        a = Variable(f3)
        test_object(a, (dim*dim+dim+1,), ())
        
        #a = MultiIndex()
        
        # --- Non-terminals:
        
        #a = Indexed()
        a = v2[i,j]
        test_object(a, (), (i,j))
        a = v2[0,k]
        test_object(a, (), (k,))
        a = v2[l,1]
        test_object(a, (), (l,))
        a = f2[i,j]
        test_object(a, (), (i,j))
        a = f2[0,k]
        test_object(a, (), (k,))
        a = f2[l,1]
        test_object(a, (), (l,))
        
        a = Inverse(I)
        test_object(a, (dim,dim), ())
        a = Inverse(v2)
        test_object(a, (dim,dim), ())
        a = Inverse(f2)
        test_object(a, (dim,dim), ())
        
        for v in (v0,v1,v2,v3):
            for f in (f0,f1,f2,f3):
                a = Outer(v, f)
                test_object(a, None, None)
        
        for v,f in zip((v0,v1,v2,v3), (f0,f1,f2,f3)):
            a = Inner(v, f)
            test_object(a, None, None)
        
        for v in (v1,v2,v3):
            for f in (f1,f2,f3):
                a = Dot(v, f)
                test_object(a, None, None)
        
        a = Cross(v1, f1)
        test_object(a, (3,), ())
        
        #a = Sum()
        a = v0 + f0 + v0
        test_object(a, (), ())
        a = v1 + f1 + v1
        test_object(a, (dim,), ())
        a = v2 + f2 + v2
        test_object(a, (dim,dim), ())
        #a = Product()
        a = 3*v0*(2.0*v0)*f0*(v0*3.0)
        test_object(a, (), ())
        #a = Division()
        a = v0 / 2.0
        test_object(a, (), ())
        a = v0 / f0
        test_object(a, (), ())
        a = v0 / (f0 + 7)
        test_object(a, (), ())
        #a = Power()
        a = f0**3
        test_object(a, (), ())
        a = (f0*2)**1.23
        test_object(a, (), ())
        
        #a = ListTensor()
        a = Vector([1.0, 2.0*f0, f0**2])
        test_object(a, (3,), ())
        a = Matrix([[1.0, 2.0*f0, f0**2],
                    [1.0, 2.0*f0, f0**2]])
        test_object(a, (2,3), ())
        a = Tensor([ [[0.00, 0.01, 0.02],
                      [0.10, 0.11, 0.12]],
                     [[1.00, 1.01, 1.02],
                      [1.10, 1.11, 1.12]] ])
        test_object(a, (2,2,3), ())
        
        #a = ComponentTensor()
        a = Vector(v1[i]*f1[j], i)
        test_object(a, (DefaultDim,), (j,))
        a = Matrix(v1[i]*f1[j], (j,i))
        test_object(a, (DefaultDim,DefaultDim), ())
        a = Tensor(v1[i]*f1[j], (i,j))
        test_object(a, (DefaultDim,DefaultDim), ())
        a = Tensor(v2[i,j]*f2[j,k], (i,k))
        test_object(a, (DefaultDim,DefaultDim), ())
        
        a = Deviatoric(v2)
        test_object(a, (dim,dim), ())
        a = Deviatoric(f2)
        test_object(a, (dim,dim), ())
        a = Deviatoric(f2*f0+v2*3)
        test_object(a, (dim,dim), ())
        a = Transposed(v2)
        test_object(a, (dim,dim), ())
        a = Transposed(f2)
        test_object(a, (dim,dim), ())
        a = Transposed(f2*f0+v2*3)
        test_object(a, (dim,dim), ())
        a = Determinant(v2)
        test_object(a, (), ())
        a = Determinant(f2)
        test_object(a, (), ())
        a = Determinant(f2*f0+v2*3)
        test_object(a, (), ())
        a = Trace(v2)
        test_object(a, (), ())
        a = Trace(f2)
        test_object(a, (), ())
        a = Trace(f2*f0+v2*3)
        test_object(a, (), ())
        a = Cofactor(v2)
        test_object(a, (dim,dim), ())
        a = Cofactor(f2)
        test_object(a, (dim,dim), ())
        a = Cofactor(f2*f0+v2*3)
        test_object(a, (dim,dim), ())
        
        cond1 = LE(f0, 1.0)
        cond2 = EQ(3.0, f0)
        cond3 = NE(sin(f0), cos(f0))
        cond4 = LT(sin(f0), cos(f0))
        cond5 = GE(sin(f0), cos(f0))
        cond6 = GT(sin(f0), cos(f0))
        a = Conditional(cond1, 1, 2)
        b = Conditional(cond2, f0**3, ln(f0))
        
        test_object(cond1, None, None)
        test_object(cond2, None, None)
        test_object(cond3, None, None)
        test_object(cond4, None, None)
        test_object(cond5, None, None)
        test_object(cond6, None, None)
        test_object(a, (), ())
        test_object(b, (), ())
        
        a = Abs(f0)
        test_object(a, (), ())
        a = Mod(f0, 3.0)
        test_object(a, (), ())
        a = Mod(3.0, f0)
        test_object(a, (), ())
        a = Sqrt(f0)
        test_object(a, (), ())
        a = Cos(f0)
        test_object(a, (), ())
        a = Sin(f0)
        test_object(a, (), ())
        a = Exp(f0)
        test_object(a, (), ())
        a = Ln(f0)
        test_object(a, (), ())
        
        a = Abs(1.0)
        test_object(a, (), ())
        a = Sqrt(1.0)
        test_object(a, (), ())
        a = Cos(1.0)
        test_object(a, (), ())
        a = Sin(1.0)
        test_object(a, (), ())
        a = Exp(1.0)
        test_object(a, (), ())
        a = Ln(1.0)
        test_object(a, (), ())
        
        # TODO:
        
        #a = SpatialDerivative()
        a = f0.dx(0)
        test_object(a, (), ())
        a = f0.dx(i)
        test_object(a, (), (i,))
        a = f0.dx(i,j,1)
        test_object(a, (), (i,j))
        
        s0 = Variable(f0)
        s1 = Variable(f1)
        s2 = Variable(f2)
        f = dot(s0*s1, s2)
        test_object(s0, (), ())
        test_object(s1, (dim,), ())
        test_object(s2, (dim,dim), ())
        test_object(f, (dim,), ())
        
        a = Diff(f, s0)
        test_object(a, (dim,), ())
        a = Diff(f, s1)
        test_object(a, (dim,dim,), ())
        a = Diff(f, s2)
        test_object(a, (dim,dim,dim), ())
        
        a = Div(v1)
        test_object(a, (), ())
        a = Div(f1)
        test_object(a, (), ())
        a = Div(v2)
        test_object(a, (dim,), ())
        a = Div(f2)
        test_object(a, (dim,), ())
        a = Div(Outer(f1,f1))
        test_object(a, (dim,), ())
        
        a = Grad(v0)
        test_object(a, (DefaultDim,), ())
        a = Grad(f0)
        test_object(a, (DefaultDim,), ())
        a = Grad(v1)
        test_object(a, (DefaultDim, dim), ())
        a = Grad(f1)
        test_object(a, (DefaultDim, dim), ())
        a = Grad(f0*v0)
        test_object(a, (DefaultDim,), ())
        a = Grad(f0*v1)
        test_object(a, (DefaultDim, dim), ())

        a = Curl(v1)
        test_object(a, (DefaultDim,), ())
        a = Curl(f1)
        test_object(a, (DefaultDim,), ())
        a = Rot(v1)
        test_object(a, (), ())
        a = Rot(f1)
        test_object(a, (), ())
        
        a = PositiveRestricted(v0)
        test_object(a, (), ())
        a = v0('+')
        test_object(a, (), ())
        a = v0('+')*f0
        test_object(a, (), ())
        
        a = NegativeRestricted(v0)
        test_object(a, (), ())
        a = v0('-')
        test_object(a, (), ())
        a = v0('-') + f0
        test_object(a, (), ())

        # --- Integrals:

        a = v0*dx
        test_form(a)
        a = v0*dx(0)
        test_form(a)
        a = v0*dx(1)
        test_form(a)
        a = v0*ds
        test_form(a)
        a = v0*ds(0)
        test_form(a)
        a = v0*ds(1)
        test_form(a)
        a = v0*dS
        test_form(a)
        a = v0*dS(0)
        test_form(a)
        a = v0*dS(1)
        test_form(a)
        
        a = v0*dot(v1,f1)*dx
        test_form(a)
        a = v0*dot(v1,f1)*dx(0)
        test_form(a)
        a = v0*dot(v1,f1)*dx(1)
        test_form(a)
        a = v0*dot(v1,f1)*ds
        test_form(a)
        a = v0*dot(v1,f1)*ds(0)
        test_form(a)
        a = v0*dot(v1,f1)*ds(1)
        test_form(a)
        a = v0*dot(v1,f1)*dS
        test_form(a)
        a = v0*dot(v1,f1)*dS(0)
        test_form(a)
        a = v0*dot(v1,f1)*dS(1)
        test_form(a)
        
        # --- Form transformations:
        
        a = f0*v0*dx + f0*v0*dot(f1,v1)*dx
        b = lhs(a)
        c = rhs(a)
        d = derivative(a, f1, v1)
        f = action(d)
        #e = action(b)
        
if __name__ == "__main__":
    unittest.main()

