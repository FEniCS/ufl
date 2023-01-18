#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

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
    assert len(compute_form_data(L).arguments) == 1
    assert len(compute_form_data(a).arguments) == 2
    assert len(compute_form_data(b).arguments) == 2


def test_divisions(self):
    element = FiniteElement("CG", triangle, 1)
    f = Coefficient(element)
    g = Coefficient(element)

    # Test simplification of division by 1
    a = f
    b = f/1
    assert a == b

    # Test simplification of division by 1.0
    a = f
    b = f/1.0
    assert a == b

    # Test simplification of division by of zero by something
    a = 0/f
    b = 0*f
    assert a == b

    # Test simplification of division by self (this simplification has been disabled)
    # a = f/f
    # b = 1
    # assert a == b


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
    # assert f*f == f**2


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
    # assert f + f == 2 * f

    # Test reordering of operands and collapsing sum
    a = f + g + f  # not collapsed, but ordered
    b = g + f + f  # not collapsed, but ordered
    c = (g + f) + f  # not collapsed, but ordered
    d = f + (f + g)  # not collapsed, but ordered
    assert a == b
    assert a == c
    assert a == d

    # Test reordering of operands and collapsing sum
    a = f + f + g  # collapsed
    b = g + (f + f)  # collapsed
    assert a == b


def test_mathfunctions(self):
    for i in (0.1, 0.3, 0.9):
        assert math.sin(i) == sin(i)
        assert math.cos(i) == cos(i)
        assert math.tan(i) == tan(i)
        assert math.sinh(i) == sinh(i)
        assert math.cosh(i) == cosh(i)
        assert math.tanh(i) == tanh(i)
        assert math.asin(i) == asin(i)
        assert math.acos(i) == acos(i)
        assert math.atan(i) == atan(i)
        assert math.exp(i) == exp(i)
        assert math.log(i) == ln(i)
        # TODO: Implement automatic simplification of conditionals?
        assert i == float(max_value(i, i-1))
        # TODO: Implement automatic simplification of conditionals?
        assert i == float(min_value(i, i+1))


def test_indexing(self):
    u = VectorConstant(triangle)
    v = VectorConstant(triangle)

    A = outer(u, v)
    A2 = as_tensor(A[i, j], (i, j))
    assert A2 == A

    Bij = u[i]*v[j]
    Bij2 = as_tensor(Bij, (i, j))[i, j]
    Bij3 = as_tensor(Bij, (i, j))
    assert Bij2 == Bij
