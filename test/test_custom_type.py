# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#

import pytest
from utils import LagrangeElement

from ufl import Mesh, triangle
from ufl.algebra import Product
from ufl.algorithms.analysis import extract_constants, has_exact_type
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.constant import Constant
from ufl.core.ufl_type import ufl_type
from ufl.corealg.multifunction import MultiFunction
from ufl.geometry import SpatialCoordinate
from ufl.operators import diff


@ufl_type
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
    # assert ab.ufl_operands == (a, b)

    assert apply_algebra_lowering(ab) == ab

    x = SpatialCoordinate(domain)

    a_dx = diff(a, x)
    assert a_dx == 0
    assert a_dx.ufl_shape == (2,)

    # TODO: expression does not get simplified, so can't check here for ax_dx == a
    ax_dx = diff(a * x, x)
    assert has_exact_type(ax_dx, LabeledConstant)
    assert extract_constants(ax_dx)[0].label == "a"


class PlainConstant(Constant):
    """Not registered, so UFL treats it as a Constant."""


@ufl_type()
class ParenthesizedConstant(Constant):
    pass


@ufl_type(handler_name="renamed_constant_handler")
class RenamedConstant(Constant):
    pass


class ConstantDispatch(MultiFunction):
    def constant(self, o):
        return "constant"

    def parenthesized_constant(self, o):
        return "parenthesized"

    def renamed_constant_handler(self, o):
        return "renamed"


def test_unregistered_subclass_is_transparent():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    assert PlainConstant._ufl_typecode_ == Constant._ufl_typecode_
    assert PlainConstant._ufl_handler_name_ == "constant"
    assert ConstantDispatch()(PlainConstant(domain)) == "constant"


def test_registered_subclass_dispatch():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    typecodes = {c._ufl_typecode_ for c in (Constant, ParenthesizedConstant, RenamedConstant)}
    assert len(typecodes) == 3
    assert ParenthesizedConstant._ufl_handler_name_ == "parenthesized_constant"
    assert RenamedConstant._ufl_handler_name_ == "renamed_constant_handler"

    dispatch = ConstantDispatch()
    assert dispatch(Constant(domain)) == "constant"
    assert dispatch(ParenthesizedConstant(domain)) == "parenthesized"
    assert dispatch(RenamedConstant(domain)) == "renamed"


def test_register_twice_raises():
    with pytest.raises(ValueError, match="already registered"):
        ufl_type(LabeledConstant)
