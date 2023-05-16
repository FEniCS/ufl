#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest

from ufl import *
from ufl.algorithms import tree_format
from ufl.algorithms.apply_derivatives import (GateauxDerivativeRuleset, GenericDerivativeRuleset, GradRuleset,
                                              VariableRuleset, apply_derivatives)
from ufl.algorithms.renumbering import renumber_indices
from ufl.classes import Grad
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1, L2
from ufl.tensors import as_tensor

# Note: the old tests in test_automatic_differentiation.py are a bit messy
#       but still cover many things that are not in here yet.


# FIXME: Write UNIT tests for all terminal derivatives!
# FIXME: Write UNIT tests for operator derivatives!


def test_apply_derivatives_doesnt_change_expression_without_derivatives():
    cell = triangle
    d = cell.geometric_dimension()
    V0 = FiniteElement("Discontinuous Lagrange", cell, 0, (), (), "identity", L2)
    V1 = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)

    # Literals
    z = zero((3, 2))
    one = as_ufl(1)
    two = as_ufl(2.0)
    I = Identity(d)
    literals = [z, one, two, I]

    # Geometry
    x = SpatialCoordinate(cell)
    n = FacetNormal(cell)
    volume = CellVolume(cell)
    geometry = [x, n, volume]

    # Arguments
    v0 = TestFunction(V0)
    v1 = TestFunction(V1)
    arguments = [v0, v1]

    # Coefficients
    f0 = Coefficient(V0)
    f1 = Coefficient(V1)
    coefficients = [f0, f1]

    # Expressions
    e0 = f0 + f1
    e1 = v0 * (f1/3 - f0**2)
    e2 = exp(sin(cos(tan(ln(x[0])))))
    expressions = [e0, e1, e2]

    # Check that all are unchanged
    for expr in literals + geometry + arguments + coefficients + expressions:
        # Note the use of "is" here instead of ==, this property
        # is important for efficiency and memory usage
        assert apply_derivatives(expr) is expr


def test_literal_derivatives_are_zero():
    cell = triangle
    d = cell.geometric_dimension()

    # Literals
    one = as_ufl(1)
    two = as_ufl(2.0)
    I = Identity(d)
    E = PermutationSymbol(d)
    literals = [one, two, I]

    # Generic ruleset handles literals directly:
    for l in literals:
        for sh in [(), (d,), (d, d+1)]:
            assert GenericDerivativeRuleset(sh)(l) == zero(l.ufl_shape + sh)

    # Variables
    v0 = variable(one)
    v1 = variable(zero((d,)))
    v2 = variable(I)
    variables = [v0, v1, v2]

    # Test literals via apply_derivatives and variable ruleset:
    for l in literals:
        for v in variables:
            assert apply_derivatives(diff(l, v)) == zero(l.ufl_shape + v.ufl_shape)

    V0 = FiniteElement("Discontinuous Lagrange", cell, 0, (), (), "identity", L2)
    V1 = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    u0 = Coefficient(V0)
    u1 = Coefficient(V1)
    v0 = TestFunction(V0)
    v1 = TestFunction(V1)
    args = [(u0, v0), (u1, v1)]

    # Test literals via apply_derivatives and variable ruleset:
    for l in literals:
        for u, v in args:
            assert apply_derivatives(derivative(l, u, v)) == zero(l.ufl_shape + v.ufl_shape)

    # Test grad ruleset directly since grad(literal) is invalid:
    assert GradRuleset(d)(one) == zero((d,))
    assert GradRuleset(d)(one) == zero((d,))


def test_grad_ruleset():
    cell = triangle
    d = cell.geometric_dimension()

    V0 = FiniteElement("Discontinuous Lagrange", cell, 0, (), (), "identity", L2)
    V1 = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    V2 = FiniteElement("Lagrange", cell, 2, (), (), "identity", H1)
    W0 = FiniteElement("Discontinuous Lagrange", cell, 0, (2, ), (2, ), "identity", L2)
    W1 = FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1)
    W2 = FiniteElement("Lagrange", cell, 2, (d, ), (d, ), "identity", H1)

    # Literals
    one = as_ufl(1)
    two = as_ufl(2.0)
    I = Identity(d)
    literals = [one, two, I]

    # Geometry
    x = SpatialCoordinate(cell)
    n = FacetNormal(cell)
    volume = CellVolume(cell)
    geometry = [x, n, volume]

    # Arguments
    u0 = TestFunction(V0)
    u1 = TestFunction(V1)
    arguments = [u0, u1]

    # Coefficients
    r = Constant(cell)
    vr = VectorConstant(cell)
    f0 = Coefficient(V0)
    f1 = Coefficient(V1)
    f2 = Coefficient(V2)
    vf0 = Coefficient(W0)
    vf1 = Coefficient(W1)
    vf2 = Coefficient(W2)
    coefficients = [f0, f1, vf0, vf1]

    # Expressions
    e0 = f0 + f1
    e1 = u0 * (f1/3 - f0**2)
    e2 = exp(sin(cos(tan(ln(x[0])))))
    expressions = [e0, e1, e2]

    # Variables
    v0 = variable(one)
    v1 = variable(f1)
    v2 = variable(f0*f1)
    variables = [v0, v1, v2]

    rules = GradRuleset(d)

    # Literals
    assert rules(one) == zero((d,))
    assert rules(two) == zero((d,))
    assert rules(I) == zero((d, d, d))

    # Assumed piecewise constant geometry
    for g in [n, volume]:
        assert rules(g) == zero(g.ufl_shape + (d,))

    # Non-constant geometry
    assert rules(x) == I

    # Arguments
    for u in arguments:
        assert rules(u) == grad(u)

    # Piecewise constant coefficients (Constant)
    assert rules(r) == zero((d,))
    assert rules(vr) == zero((d, d))
    assert rules(grad(r)) == zero((d, d))
    assert rules(grad(vr)) == zero((d, d, d))

    # Piecewise constant coefficients (DG0)
    assert rules(f0) == zero((d,))
    assert rules(vf0) == zero((d, d))
    assert rules(grad(f0)) == zero((d, d))
    assert rules(grad(vf0)) == zero((d, d, d))

    # Piecewise linear coefficients
    assert rules(f1) == grad(f1)
    assert rules(vf1) == grad(vf1)
    # assert rules(grad(f1)) == zero((d,d)) # TODO: Use degree to make this work
    # assert rules(grad(vf1)) == zero((d,d,d))

    # Piecewise quadratic coefficients
    assert rules(grad(f2)) == grad(grad(f2))
    assert rules(grad(vf2)) == grad(grad(vf2))

    # Indexed coefficients
    assert renumber_indices(apply_derivatives(grad(vf2[0]))) == renumber_indices(grad(vf2)[0, :])
    assert renumber_indices(apply_derivatives(grad(vf2[1])[0])) == renumber_indices(grad(vf2)[1, 0])

    # Grad of gradually more complex expressions
    assert apply_derivatives(grad(2*f0)) == zero((d,))
    assert renumber_indices(apply_derivatives(grad(2*f1))) == renumber_indices(2*grad(f1))
    assert renumber_indices(apply_derivatives(grad(sin(f1)))) == renumber_indices(cos(f1) * grad(f1))
    assert renumber_indices(apply_derivatives(grad(cos(f1)))) == renumber_indices(-sin(f1) * grad(f1))


def test_variable_ruleset():
    pass


def test_gateaux_ruleset():
    pass
