"""Tests for balancing terminal modifiers."""

from utils import LagrangeElement

from ufl import FunctionSpace, Mesh, TestFunction, grad, triangle
from ufl.algorithms.balancing import balance_modifiers
from ufl.classes import Grad, PositiveRestricted


def test_balance_modifiers_preserves_precedence_and_identity():
    """Move restrictions outside derivatives and reuse balanced expressions."""
    mesh = Mesh(LagrangeElement(triangle, 1, (2,)))
    space = FunctionSpace(mesh, LagrangeElement(triangle, 1))
    u = TestFunction(space)

    unbalanced = grad(PositiveRestricted(u))
    balanced = balance_modifiers(unbalanced)

    assert balanced == PositiveRestricted(Grad(u))
    assert balance_modifiers(balanced) is balanced
