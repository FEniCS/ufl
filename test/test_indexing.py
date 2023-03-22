#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest
from ufl_legacy import *
from ufl_legacy.classes import *


@pytest.fixture
def x1():
    x = SpatialCoordinate(triangle)
    return x


@pytest.fixture
def x2():
    x = SpatialCoordinate(triangle)
    return outer(x, x)


@pytest.fixture
def x3():
    x = SpatialCoordinate(triangle)
    return outer(outer(x, x), x)


def test_annotated_literals():
    z = Zero(())
    assert z.ufl_shape == ()
    assert z.ufl_free_indices == ()
    assert z.ufl_index_dimensions == ()
    #assert z.free_indices() == ()  # Deprecated interface
    #assert z.index_dimensions() == {}  # Deprecated interface

    z = Zero((3,))
    assert z.ufl_shape == (3,)
    assert z.ufl_free_indices == ()
    assert z.ufl_index_dimensions == ()
    #assert z.free_indices() == ()  # Deprecated interface
    #assert z.index_dimensions() == {}  # Deprecated interface

    i = Index(count=2)
    j = Index(count=4)
    # z = Zero((), (2, 4), (3, 5))
    z = Zero((), (j, i), {i: 3, j: 5})
    assert z.ufl_shape == ()
    #assert z.free_indices() == (i, j)  # Deprecated interface
    #assert z.index_dimensions() == {i: 3, j: 5}  # Deprecated interface
    assert z.ufl_free_indices == (2, 4)
    assert z.ufl_index_dimensions == (3, 5)


def test_fixed_indexing_of_expression(x1, x2, x3):
    x0 = x1[0]
    x00 = x2[0, 0]
    x000 = x3[0, 0, 0]
    assert isinstance(x0, Indexed)
    assert isinstance(x00, Indexed)
    assert isinstance(x000, Indexed)
    assert isinstance(x0.ufl_operands[0], SpatialCoordinate)
    assert isinstance(x00.ufl_operands[0], Outer)
    assert isinstance(x000.ufl_operands[0], Outer)
    assert isinstance(x0.ufl_operands[1], MultiIndex)
    assert isinstance(x00.ufl_operands[1], MultiIndex)
    assert isinstance(x000.ufl_operands[1], MultiIndex)

    mi = x000.ufl_operands[1]
    assert len(mi) == 3
    assert mi.indices() == (FixedIndex(0),) * 3


def test_indexed():
    pass


def test_indexsum():
    pass


def test_componenttensor():
    pass


def test_tensoralgebra():
    pass
