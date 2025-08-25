"""Tests of function spaces."""

import pytest
from utils import LagrangeElement

from ufl import FunctionSpace, Mesh, quadrilateral, triangle


def test_cell_mismatch():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    elements = LagrangeElement(quadrilateral, 1)

    with pytest.raises(ValueError):
        FunctionSpace(domain, elements)


def test_wrong_order():
    domain = Mesh(
        [
            LagrangeElement(quadrilateral, 1, (2,)),
            LagrangeElement(triangle, 1, (2,)),
        ]
    )
    elements = [LagrangeElement(triangle, 1), LagrangeElement(quadrilateral, 1)]

    with pytest.raises(ValueError):
        FunctionSpace(domain, elements)
