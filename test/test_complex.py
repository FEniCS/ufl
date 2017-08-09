#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest
from ufl import *
from ufl.constantvalue import Zero, ComplexValue, FloatValue
from ufl.algebra import Conj, Real, Imag
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms import estimate_total_polynomial_degree
# from ufl.algorithms.comparison_checker import do_comparison_check


def test_conj(self):
	z1 = ComplexValue(1+2j)
	z2 = ComplexValue(1-2j)

	assert z1 == Conj(z2)
	assert z2 == Conj(z1)


def test_real(self):
	z0 = Zero()
	z1 = as_ufl(1.0)
	z2 = ComplexValue(1j)
	z3 = ComplexValue(1+1j)

	assert Real(z1) == z1
	assert Real(z3) == z1
	assert Real(z2) == z0


def test_imag(self):
	z0 = Zero()
	z1 = as_ufl(1.0)
	z2 = as_ufl(1j)
	z3 = ComplexValue(1+1j)

	assert Imag(z2) == z1
	assert Imag(z3) == z1
	assert Imag(z1) == z0


def test_apply_algebra_lowering_complex(self):
	cell = triangle
	element = FiniteElement("Lagrange", cell, 1)

	v = TestFunction(element)
	u = TrialFunction(element)

	a = dot(u, v)
	b = inner(u, v)
	c = outer(u, v)

	assert apply_algebra_lowering(a) == u*conj(v)
	assert apply_algebra_lowering(b) == u*conj(v)
	assert apply_algebra_lowering(c) == v*conj(u)


def test_remove_complex_nodes(self):
	cell = triangle
	element = FiniteElement("Lagrange", cell, 1)

	u = TrialFunction(element)
	v = TestFunction(element)
	f = Coefficient(element)

	a = conj(v)
	b = real(u)
	c = imag(f)
	d = conj(real(v))*imag(conj(u))

	assert remove_complex_nodes(a) == v
	assert remove_complex_nodes(b) == u
	assert remove_complex_nodes(c) == f
	assert remove_complex_nodes(d) == u*v


# def test_comparison_checker(self):
# 	cell = triangle
# 	element = FiniteElement("Lagrange", cell, 1)

# 	u = TrialFunction(element)
# 	v = TestFunction(element)

# 	a = conditional(ge(cc,cc),cc*dot(grad(v),grad(u)),dot(grad(v),grad(u)))

# 	assert do_comparison_check(a)
	
# def test_complex_arities(self):


def test_complex_degree_handling(self):
	cell = triangle
	element = FiniteElement("Lagrange", cell, 3)

	v = TestFunction(element)

	a = conj(v)
	b = imag(v)
	c = real(v)

	# complex operators don't change the degree of a polynomial
	assert estimate_total_polynomial_degree(a) == 3
	assert estimate_total_polynomial_degree(b) == 3
	assert estimate_total_polynomial_degree(c) == 3


# def test_complex_differentiation_rules(self):


