# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#

from utils import LagrangeElement

from ufl import Mesh, triangle
from ufl.algebra import Product
from ufl.algorithms.analysis import extract_constants, has_exact_type
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.constant import Constant
from ufl.core.ufl_type import ufl_type
from ufl.geometry import SpatialCoordinate
from ufl.operators import diff


@ufl_type()
class LabeledConstant(Constant):
    def __init__(self, domain, shape=(), count=None, label: str = "c"):
        Constant.__init__(self, domain, shape, count)
        self._label = label

    @property
    def label(self) -> str:
        return self._label


def test():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    a = LabeledConstant(domain, label="a")
    b = LabeledConstant(domain, label="b")

    assert a.label == "a"
    assert b.label == "b"

    ab = a * b
    assert isinstance(ab, Product)
    assert ab.ufl_operands == (a, b)

    assert apply_algebra_lowering(ab) == ab

    x = SpatialCoordinate(domain)

    a_dx = diff(a, x)
    assert a_dx == 0
    assert a_dx.ufl_shape == (2,)

    # TODO: expression does not get simplified, so can't check here for ax_dx == a
    ax_dx = diff(a * x, x)
    assert has_exact_type(ax_dx, LabeledConstant)
    assert extract_constants(ax_dx)[0].label == "a"
