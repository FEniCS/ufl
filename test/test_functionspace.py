"""Tests of function spaces."""

import pytest
import utils  # noqa: F401
from mockobjects import MockMesh
from utils import FiniteElement, LagrangeElement

import ufl  # noqa: F401
from ufl import (
    Cell,
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dS,
    ds,
    dx,
    hexahedron,
    interval,
    quadrilateral,
    tetrahedron,
    triangle,
)
from ufl.algorithms import compute_form_data
from ufl.domain import extract_domains
from ufl.pullback import (
    IdentityPullback,  # noqa: F401
    identity_pullback,
)
from ufl.sobolevspace import H1


def test_cell_mismatch():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    elements = LagrangeElement(quadrilateral, 1)

    with pytest.raises(ValueError):
        space = FunctionSpace(domain, elements)


def test_wrong_order():
    domain = Mesh(
        [
            LagrangeElement(quadrilateral, 1, (2,)),
            LagrangeElement(triangle, 1, (2,)),
        ]
    )
    elements = [LagrangeElement(triangle, 1), LagrangeElement(quadrilateral, 1)]

    with pytest.raises(ValueError):
        space = FunctionSpace(domain, elements)
