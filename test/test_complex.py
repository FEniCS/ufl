#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest
import cmath
from ufl.constantvalue import Zero, ComplexValue
from ufl.algebra import Conj, Real, Imag
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms.optimise_complex_nodes import optimise_complex_nodes
from ufl.algorithms import estimate_total_polynomial_degree
from ufl.algorithms.comparison_checker import do_comparison_check, ComplexComparisonError
from ufl.algorithms.formtransformations import compute_form_adjoint
from ufl import TestFunction, TrialFunction, triangle, FiniteElement, \
        as_ufl, inner, grad, dx, dot, outer, conj, sqrt, sin, cosh, \
        atan, ln, exp, as_tensor, real, imag, conditional, \
        min_value, max_value, gt, lt, cos, ge, le, Coefficient


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


def test_compute_form_adjoint(self):
    cell = triangle
    element = FiniteElement('Lagrange', cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)

    a = inner(grad(u), grad(v)) * dx

    assert compute_form_adjoint(a) == conj(inner(grad(v), grad(u))) * dx


def test_complex_algebra(self):
    z1 = ComplexValue(1j)
    z2 = ComplexValue(1+1j)

    # Remember that ufl.algebra functions return ComplexValues, but ufl.mathfunctions return complex Python scalar
    # Any operations with a ComplexValue and a complex Python scalar promote to ComplexValue
    assert z1*z2 == ComplexValue(-1+1j)
    assert z2/z1 == ComplexValue(1-1j)
    assert pow(z2, z1) == ComplexValue((1+1j)**1j)
    assert sqrt(z2) * as_ufl(1) == ComplexValue(cmath.sqrt(1+1j))
    assert ((sin(z2) + cosh(z2) - atan(z2)) * z1) == ComplexValue((cmath.sin(1+1j) + cmath.cosh(1+1j) - cmath.atan(1+1j))*1j)
    assert (abs(z2) - ln(z2))/exp(z1) == ComplexValue((abs(1+1j) - cmath.log(1+1j))/cmath.exp(1j))


def test_automatic_simplification(self):
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    v = TestFunction(element)
    u = TrialFunction(element)

    assert inner(u, v) == u * conj(v)
    assert dot(u, v) == u * conj(v)
    assert outer(u, v) == conj(u) * v


def test_apply_algebra_lowering_complex(self):
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    v = TestFunction(element)
    u = TrialFunction(element)

    gv = grad(v)
    gu = grad(u)

    a = dot(gu, gv)
    b = inner(gu, gv)
    c = outer(gu, gv)

    lowered_a = apply_algebra_lowering(a)
    lowered_b = apply_algebra_lowering(b)
    lowered_c = apply_algebra_lowering(c)
    lowered_a_index = lowered_a.index()
    lowered_b_index = lowered_b.index()
    lowered_c_indices = lowered_c.indices()

    assert lowered_a == gu[lowered_a_index] * conj(gv[lowered_a_index])
    assert lowered_b == gu[lowered_b_index] * conj(gv[lowered_b_index])
    assert lowered_c == as_tensor(conj(gu[lowered_c_indices[0]]) * gv[lowered_c_indices[1]], (lowered_c_indices[0],) + (lowered_c_indices[1],))


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


def test_optimise_complex_nodes(self):
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)

    a = conj(conj(u))
    b = conj(as_ufl(0))
    c = abs(conj(u))
    d = abs(as_ufl(0))
    e = conj(abs(u))
    f = conj(real(u))
    g = conj(imag(u))
    h = abs(abs(u))
    i = real(real(u))
    j = real(conj(u))
    k = imag(imag(u))
    l = imag(real(u))
    m = imag(abs(u))
    n = imag(as_ufl(0))
    o = real(as_ufl(0))

    assert optimise_complex_nodes(a) == u
    assert optimise_complex_nodes(b) == 0
    assert optimise_complex_nodes(c) == abs(u)
    assert optimise_complex_nodes(d) == 0
    assert optimise_complex_nodes(e) == abs(u)
    assert optimise_complex_nodes(f) == real(u)
    assert optimise_complex_nodes(g) == imag(u)
    assert optimise_complex_nodes(h) == abs(u)
    assert optimise_complex_nodes(i) == real(u)
    assert optimise_complex_nodes(j) == real(u)
    assert optimise_complex_nodes(k) == 0
    assert optimise_complex_nodes(l) == 0
    assert optimise_complex_nodes(m) == 0
    assert optimise_complex_nodes(n) == 0
    assert optimise_complex_nodes(o) == 0


def test_comparison_checker(self):
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)

    u = TrialFunction(element)
    v = TestFunction(element)

    a = conditional(ge(abs(u), imag(v)), u, v)
    b = conditional(le(sqrt(abs(u)), imag(v)), as_ufl(1), as_ufl(1j))
    c = conditional(gt(abs(u), pow(imag(v), 0.5)), sin(u), cos(v))
    d = conditional(lt(as_ufl(-1), as_ufl(1)), u, v)
    e = max_value(as_ufl(0), real(u))
    f = min_value(sin(u), cos(v))
    g = min_value(sin(pow(u, 3)), cos(abs(v)))

    assert do_comparison_check(a) == conditional(ge(real(abs(u)), real(imag(v))), u, v)
    with pytest.raises(ComplexComparisonError):
        b = do_comparison_check(b)
    with pytest.raises(ComplexComparisonError):
        c = do_comparison_check(c)
    assert do_comparison_check(d) == conditional(lt(real(as_ufl(-1)), real(as_ufl(1))), u, v)
    assert do_comparison_check(e) == max_value(real(as_ufl(0)), real(real(u)))
    assert do_comparison_check(f) == min_value(real(sin(u)), real(cos(v)))
    assert do_comparison_check(g) == min_value(real(sin(pow(u, 3))), real(cos(abs(v))))


def test_complex_degree_handling(self):
    cell = triangle
    element = FiniteElement("Lagrange", cell, 3)

    v = TestFunction(element)

    a = conj(v)
    b = imag(v)
    c = real(v)

    assert estimate_total_polynomial_degree(a) == 3
    assert estimate_total_polynomial_degree(b) == 3
    assert estimate_total_polynomial_degree(c) == 3
