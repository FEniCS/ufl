#!/usr/bin/env py.test

import pytest
from ufl.classes import Sum, Product
import math
from ufl import *

def xtest_zero_times_argument(self):
    # FIXME: Allow zero forms
    element = FiniteElement("CG", triangle, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    L = 0*v*dx
    a = 0*(u*v)*dx
    b = (0*u)*v*dx
    self.assertEqual(len(compute_form_data(L).arguments), 1)
    self.assertEqual(len(compute_form_data(a).arguments), 2)
    self.assertEqual(len(compute_form_data(b).arguments), 2)

def test_divisions(self):
    element = FiniteElement("CG", triangle, 1)
    f = Coefficient(element)
    g = Coefficient(element)

    # Test simplification of division by 1
    a = f
    b = f/1
    self.assertEqual(a, b)

    # Test simplification of division by 1.0
    a = f
    b = f/1.0
    self.assertEqual(a, b)

    # Test simplification of division by of zero by something
    a = 0/f
    b = 0*f
    self.assertEqual(a, b)

    # Test simplification of division by self (this simplification has been disabled)
    #a = f/f
    #b = 1
    #self.assertEqual(a, b)

def test_products(self):
    element = FiniteElement("CG", triangle, 1)
    f = Coefficient(element)
    g = Coefficient(element)

    # Test simplification of literal multiplication
    assert f*0 == as_ufl(0)
    assert 0*f == as_ufl(0)
    assert 1*f == f
    assert f*1 == f
    assert as_ufl(2)*as_ufl(3) == as_ufl(6)
    assert as_ufl(2.0)*as_ufl(3.0) == as_ufl(6.0)

    # Test reordering of operands
    assert f*g == g*f

    # Test simplification of self-multiplication (this simplification has been disabled)
    #assert f*f == f**2

def test_sums(self):
    element = FiniteElement("CG", triangle, 1)
    f = Coefficient(element)
    g = Coefficient(element)

    # Test reordering of operands
    assert f + g == g + f

    # Test adding zero
    assert f + 0 == f
    assert 0 + f == f

    # Test collapsing of basic sum (this simplification has been disabled)
    #assert f + f == 2 * f

    # Test reordering of operands and collapsing sum
    a = f + g + f # not collapsed, but ordered
    b = g + f + f # not collapsed, but ordered
    c = (g + f) + f # not collapsed, but ordered
    d = f + (f + g) # not collapsed, but ordered
    self.assertEqual(a, b)
    self.assertEqual(a, c)
    self.assertEqual(a, d)

    # Test reordering of operands and collapsing sum
    a = f + f + g # collapsed
    b = g + (f + f) # collapsed
    self.assertEqual(a, b)

def test_mathfunctions(self):
    for i in (0.1, 0.3, 0.9):
        self.assertEqual(math.sin(i), sin(i))
        self.assertEqual(math.cos(i), cos(i))
        self.assertEqual(math.tan(i), tan(i))
        self.assertEqual(math.sinh(i), sinh(i))
        self.assertEqual(math.cosh(i), cosh(i))
        self.assertEqual(math.tanh(i), tanh(i))
        self.assertEqual(math.asin(i), asin(i))
        self.assertEqual(math.acos(i), acos(i))
        self.assertEqual(math.atan(i), atan(i))
        self.assertEqual(math.exp(i), exp(i))
        self.assertEqual(math.log(i), ln(i))
        # TODO: Implement automatic simplification of conditionals?
        self.assertEqual(i, float(Max(i, i-1)))
        # TODO: Implement automatic simplification of conditionals?
        self.assertEqual(i, float(Min(i, i+1)))

def test_indexing(self):
    u = VectorConstant(triangle)
    v = VectorConstant(triangle)

    A = outer(u, v)
    A2 = as_tensor(A[i, j], (i, j))
    self.assertEqual(A2, A)

    Bij = u[i]*v[j]
    Bij2 = as_tensor(Bij, (i, j))[i, j]
    Bij3 = as_tensor(Bij, (i, j))
    #print "BIJ:  ", Bij.free_indices()
    #print "BIJ2: ", Bij2.free_indices()
    #print "BIJ:  ", Bij
    #print "BIJ2: ", Bij2
    self.assertEqual(Bij2, Bij)
