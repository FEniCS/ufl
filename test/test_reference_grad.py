"""Tests for the user-facing ``ufl.reference_grad`` wrapper."""

from utils import LagrangeElement

import ufl
from ufl import Coefficient, FunctionSpace, Mesh, interval, reference_grad, triangle
from ufl.classes import ReferenceGrad, ReferenceValue


def test_reference_grad_is_exported():
    assert ufl.reference_grad is reference_grad
    assert "reference_grad" in ufl.__all__


def test_reference_grad_returns_reference_grad():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    f = Coefficient(V)

    expr = reference_grad(ReferenceValue(f))

    assert isinstance(expr, ReferenceGrad)
    assert expr == ReferenceGrad(ReferenceValue(f))


def test_reference_grad_shape_appends_topological_dimension():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    f = Coefficient(V)

    expr = reference_grad(ReferenceValue(f))

    assert expr.ufl_shape == (2,)


def test_reference_grad_on_interval():
    domain = Mesh(LagrangeElement(interval, 1, (1,)))
    V = FunctionSpace(domain, LagrangeElement(interval, 1))
    f = Coefficient(V)

    expr = reference_grad(ReferenceValue(f))

    assert expr.ufl_shape == (1,)
