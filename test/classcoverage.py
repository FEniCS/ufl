#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-06 -- 2009-02-10"

import unittest

import ufl
from ufl import *
from ufl.constantvalue import as_ufl
from ufl.classes import * 
from ufl.algorithms import * 

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
    ce = a.cell()
    
    # Compare with provided properties
    if free_indices is not None:
        assert len(set(fi) ^ set(free_indices)) == 0
    if shape is not None:
        if sh != shape:
            print "sh:", sh
            print "shape:", shape
        assert sh == shape

def test_object2(a):
    # Test reproduction via repr string
    r = repr(a)
    e = eval(r, globals())
    assert hash(a) == hash(e)
    
    # Can't really test str more than that it exists
    s = str(a)
    
    # Check that some properties are at least available
    ce = a.cell()

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
    
    def testExports(self):
        "Verify that ufl.classes exports all Expr subclasses."
        all_expr_classes = []
        for m in vars(ufl).values():
            if isinstance(m, type(ufl)):
                for c in vars(m).values():
                    if isinstance(c, type) and issubclass(c, Expr):
                        all_expr_classes.append(c)
        missing_classes = set(c.__name__ for c in all_expr_classes) - set(c.__name__ for c in all_ufl_classes)
        if missing_classes:
            print "The following subclasses of Expr were not exported from ufl.classes:"
            print "\n".join(sorted(missing_classes))
        self.assertFalse(missing_classes)
    
    def testAll(self):
        
        # --- Elements:
        cell = triangle
        dim = cell.d
        
        e0 = FiniteElement("CG", cell, 1)
        e1 = VectorElement("CG", cell, 1)
        e2 = TensorElement("CG", cell, 1)
        e3 = MixedElement(e0, e1, e2)

        e13D = VectorElement("CG", tetrahedron, 1)
        
        # --- Terminals:
        
        v13D = Argument(e13D)
        f13D = Coefficient(e13D)
        
        v0 = Argument(e0)
        v1 = Argument(e1)
        v2 = Argument(e2)
        v3 = Argument(e3)
        
        test_object(v0, (), ())
        test_object(v1, (dim,), ())
        test_object(v2, (dim,dim), ())
        test_object(v3, (dim*dim+dim+1,), ())
        
        f0 = Coefficient(e0)
        f1 = Coefficient(e1)
        f2 = Coefficient(e2)
        f3 = Coefficient(e3)
        
        test_object(f0, (), ())
        test_object(f1, (dim,), ())
        test_object(f2, (dim,dim), ())
        test_object(f3, (dim*dim+dim+1,), ())
        
        c = Constant(cell)
        test_object(c, (), ())
        
        a = FloatValue(1.23)
        test_object(a, (), ())
        
        a = IntValue(123)
        test_object(a, (), ())
        
        I = Identity(2)
        test_object(I, (dim,dim), ())
        
        n = cell.n
        test_object(n, (dim,), ())
        
        a = variable(v0)
        test_object(a, (), ())
        a = variable(v1)
        test_object(a, (dim,), ())
        a = variable(v2)
        test_object(a, (dim,dim), ())
        a = variable(v3)
        test_object(a, (dim*dim+dim+1,), ())
        a = variable(f0)
        test_object(a, (), ())
        a = variable(f1)
        test_object(a, (dim,), ())
        a = variable(f2)
        test_object(a, (dim,dim), ())
        a = variable(f3)
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
        
        a = inv(I)
        test_object(a, (dim,dim), ())
        a = inv(v2)
        test_object(a, (dim,dim), ())
        a = inv(f2)
        test_object(a, (dim,dim), ())
        
        for v in (v0,v1,v2,v3):
            for f in (f0,f1,f2,f3):
                a = outer(v, f)
                test_object(a, None, None)
        
        for v,f in zip((v0,v1,v2,v3), (f0,f1,f2,f3)):
            a = inner(v, f)
            test_object(a, None, None)
        
        for v in (v1,v2,v3):
            for f in (f1,f2,f3):
                a = dot(v, f)
                test_object(a, None, None)
        
        a = cross(v1, f1)
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
        
        #a = Listas_tensor()
        a = as_vector([1.0, 2.0*f0, f0**2])
        test_object(a, (3,), ())
        a = as_matrix([[1.0, 2.0*f0, f0**2],
                    [1.0, 2.0*f0, f0**2]])
        test_object(a, (2,3), ())
        a = as_tensor([ [[0.00, 0.01, 0.02],
                         [0.10, 0.11, 0.12] ],
                      [ [1.00, 1.01, 1.02],
                        [1.10, 1.11, 1.12]] ])
        test_object(a, (2,2,3), ())
        
        #a = Componentas_tensor()
        a = as_vector(v1[i]*f1[j], i)
        test_object(a, (dim,), (j,))
        a = as_matrix(v1[i]*f1[j], (j,i))
        test_object(a, (dim, dim), ())
        a = as_tensor(v1[i]*f1[j], (i,j))
        test_object(a, (dim, dim), ())
        a = as_tensor(v2[i,j]*f2[j,k], (i,k))
        test_object(a, (dim, dim), ())
        
        a = dev(v2)
        test_object(a, (dim,dim), ())
        a = dev(f2)
        test_object(a, (dim,dim), ())
        a = dev(f2*f0+v2*3)
        test_object(a, (dim,dim), ())
        
        a = v2.T
        test_object(a, (dim,dim), ())
        a = f2.T
        test_object(a, (dim,dim), ())
        a = transpose(f2*f0+v2*3)
        test_object(a, (dim,dim), ())
        
        a = det(v2)
        test_object(a, (), ())
        a = det(f2)
        test_object(a, (), ())
        a = det(f2*f0+v2*3)
        test_object(a, (), ())
        
        a = tr(v2)
        test_object(a, (), ())
        a = tr(f2)
        test_object(a, (), ())
        a = tr(f2*f0+v2*3)
        test_object(a, (), ())
        
        a = cofac(v2)
        test_object(a, (dim,dim), ())
        a = cofac(f2)
        test_object(a, (dim,dim), ())
        a = cofac(f2*f0+v2*3)
        test_object(a, (dim,dim), ())
        
        cond1 = le(f0, 1.0)
        cond2 = eq(3.0, f0)
        cond3 = ne(sin(f0), cos(f0))
        cond4 = lt(sin(f0), cos(f0))
        cond5 = ge(sin(f0), cos(f0))
        cond6 = gt(sin(f0), cos(f0))
        a = conditional(cond1, 1, 2)
        b = conditional(cond2, f0**3, ln(f0))
        
        test_object2(cond1)
        test_object2(cond2)
        test_object2(cond3)
        test_object2(cond4)
        test_object2(cond5)
        test_object2(cond6)
        test_object(a, (), ())
        test_object(b, (), ())
        
        a = abs(f0)
        test_object(a, (), ())
        a = sqrt(f0)
        test_object(a, (), ())
        a = cos(f0)
        test_object(a, (), ())
        a = sin(f0)
        test_object(a, (), ())
        a = exp(f0)
        test_object(a, (), ())
        a = ln(f0)
        test_object(a, (), ())
        
        one = as_ufl(1)
        a = abs(one)
        test_object(a, (), ())
        a = Sqrt(one)
        test_object(a, (), ())
        a = Cos(one)
        test_object(a, (), ())
        a = Sin(one)
        test_object(a, (), ())
        a = Exp(one)
        test_object(a, (), ())
        a = Ln(one)
        test_object(a, (), ())
        
        # TODO:
        
        #a = SpatialDerivative()
        a = f0.dx(0)
        test_object(a, (), ())
        a = f0.dx(i)
        test_object(a, (), (i,))
        a = f0.dx(i,j,1)
        test_object(a, (), (i,j))
        
        s0 = variable(f0)
        s1 = variable(f1)
        s2 = variable(f2)
        f = dot(s0*s1, s2)
        test_object(s0, (), ())
        test_object(s1, (dim,), ())
        test_object(s2, (dim,dim), ())
        test_object(f, (dim,), ())
        
        a = diff(f, s0)
        test_object(a, (dim,), ())
        a = diff(f, s1)
        test_object(a, (dim,dim,), ())
        a = diff(f, s2)
        test_object(a, (dim,dim,dim), ())
        
        a = div(v1)
        test_object(a, (), ())
        a = div(f1)
        test_object(a, (), ())
        a = div(v2)
        test_object(a, (dim,), ())
        a = div(f2)
        test_object(a, (dim,), ())
        a = div(Outer(f1,f1))
        test_object(a, (dim,), ())
        
        a = grad(v0)
        test_object(a, (dim,), ())
        a = grad(f0)
        test_object(a, (dim,), ())
        a = grad(v1)
        test_object(a, (dim, dim), ())
        a = grad(f1)
        test_object(a, (dim, dim), ())
        a = grad(f0*v0)
        test_object(a, (dim,), ())
        a = grad(f0*v1)
        test_object(a, (dim, dim), ())

        a = curl(v13D)
        test_object(a, (3,), ())
        a = curl(f13D)
        test_object(a, (3,), ())
        a = rot(v1)
        test_object(a, (), ())
        a = rot(f1)
        test_object(a, (), ())
        
        #a = PositiveRestricted(v0)
        test_object(a, (), ())
        a = v0('+')
        test_object(a, (), ())
        a = v0('+')*f0
        test_object(a, (), ())
        
        #a = NegativeRestricted(v0)
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
        #b = lhs(a) # TODO
        #c = rhs(a) # TODO
        d = derivative(a, f1, v1)
        f = action(d)
        #e = action(b)

        # TODO: Add tests for TensorConstant, VectorConstant, ScalarSomething, Skew
        
        # --- Check which classes have been created
        if ufl.expr._class_usage_statistics:
            s = ufl.expr._class_usage_statistics
            constructed = set(s.keys())
            abstract = set((Expr, Terminal, Operator, FormArgument, AlgebraOperator,
                            Condition, MathFunction, Restricted, ScalarValue,
                            ConstantValue, CompoundDerivative, Derivative,
                            WrapperType, GeometricQuantity, CompoundTensorOperator, UtilityType))
            unused = set(ufl.classes.all_ufl_classes) - constructed - abstract
            if unused:
                print 
                print "The following classes were never instantiated in class coverage test:"
                print "\n".join(sorted(map(str,unused)))
                print 


tests = [ClasscoverageTest]

if __name__ == "__main__":
    unittest.main()

