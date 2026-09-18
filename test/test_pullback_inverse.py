"""Tests of the inverse pull backs."""

import numpy as np
import pytest
from utils import FiniteElement, LagrangeElement, MixedElement, SymmetricElement

from ufl import Cell, Coefficient, FunctionSpace, Mesh
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.cancel_jacobian_products import cancel_jacobian_products
from ufl.algorithms.remove_component_tensors import remove_component_tensors
from ufl.algorithms.renumbering import renumber_indices
from ufl.classes import JacobianDeterminant, ReferenceValue
from ufl.pullback import (
    contravariant_piola,
    covariant_contravariant_piola,
    covariant_piola,
    double_contravariant_piola,
    double_covariant_piola,
    l2_piola,
    physical_pullback,
    undefined_pullback,
)
from ufl.sobolevspace import H1, L2, HCurl, HDiv, HDivDiv, HEin

cell = Cell("triangle")
domain = Mesh(LagrangeElement(cell, 1, (2,)))

U = LagrangeElement(cell, 1)
Vd = FiniteElement("Raviart-Thomas", cell, 1, (2,), contravariant_piola, HDiv)
Vc = FiniteElement("N1curl", cell, 1, (2,), covariant_piola, HCurl)
Td = FiniteElement("Regge", cell, 1, (2, 2), double_covariant_piola, HEin)
Tc = FiniteElement("HHJ", cell, 1, (2, 2), double_contravariant_piola, HDivDiv)
Tcc = FiniteElement("CC", cell, 1, (2, 2), covariant_contravariant_piola, HDivDiv)
S = SymmetricElement({(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 2}, [U, U, U])
M = MixedElement([U, Vd, Vc])


def simplify(expr):
    """Cancel the Jacobian products a round trip leaves behind."""
    return cancel_jacobian_products(remove_component_tensors(apply_derivatives(expr)))


@pytest.mark.parametrize(
    "element",
    [U, Vd, Vc, Td, Tc, Tcc, S, M],
    ids=[
        "identity",
        "contravariant",
        "covariant",
        "double covariant",
        "double contravariant",
        "covariant contravariant",
        "symmetric",
        "mixed",
    ],
)
def test_apply_inverse_undoes_apply(element):
    """The inverse pull back returns a pushed-forward function unchanged."""
    pullback = element.pullback
    reference = ReferenceValue(Coefficient(FunctionSpace(domain, element)))
    actual = simplify(pullback.apply_inverse(pullback.apply(reference)))

    assert actual.ufl_shape == reference.ufl_shape
    for idx in np.ndindex(reference.ufl_shape):
        assert renumber_indices(actual[idx]) == renumber_indices(reference[idx])


def test_l2_piola_apply_inverse():
    """The L2 Piola scales by the Jacobian determinant.

    The round trip leaves ``detJ / detJ`` standing, because cancelling a scalar
    factor is not something ``cancel_jacobian_products`` does.
    """
    element = FiniteElement("Discontinuous Lagrange", cell, 1, (), l2_piola, L2)
    reference = ReferenceValue(Coefficient(FunctionSpace(domain, element)))

    assert l2_piola.apply_inverse(reference) == reference * JacobianDeterminant(domain)


def test_apply_inverse_of_physical_value_shape():
    """The inverse pull back maps a physical shape to a reference shape."""
    for element in [U, Vd, Vc, Td, Tc, Tcc, S, M]:
        pullback = element.pullback
        physical = pullback.apply(ReferenceValue(Coefficient(FunctionSpace(domain, element))))
        assert physical.ufl_shape == pullback.physical_value_shape(element, domain)
        assert pullback.apply_inverse(physical).ufl_shape == element.reference_value_shape


@pytest.mark.parametrize("pullback", [physical_pullback, undefined_pullback])
def test_apply_inverse_is_not_defined(pullback):
    """A pull back with no standard inverse says so."""
    element = FiniteElement("Custom", cell, 1, (), pullback, H1)
    reference = ReferenceValue(Coefficient(FunctionSpace(domain, element)))
    if pullback is undefined_pullback:
        with pytest.raises(BaseException):
            pullback.apply_inverse(reference)
    else:
        assert pullback.apply_inverse(reference) == reference
