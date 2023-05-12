#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

import pytest

from ufl_legacy import *
from ufl_legacy.algorithms import *

# TODO: Add more illegal expressions to check!


def selement():
    return FiniteElement("Lagrange", "triangle", 1)


def velement():
    return VectorElement("Lagrange", "triangle", 1)


@pytest.fixture
def a():
    return Argument(selement(), 2)


@pytest.fixture
def b():
    return Argument(selement(), 3)


@pytest.fixture
def v():
    return Argument(velement(), 4)


@pytest.fixture
def u():
    return Argument(velement(), 5)


@pytest.fixture
def f():
    return Coefficient(selement())


@pytest.fixture
def g():
    return Coefficient(selement())


@pytest.fixture
def vf():
    return Coefficient(velement())


@pytest.fixture
def vg():
    return Coefficient(velement())


def test_mul_v_u(v, u):
    with pytest.raises(UFLException):
        v * u


def test_mul_vf_u(vf, u):
    with pytest.raises(UFLException):
        vf * u


def test_mul_vf_vg(vf, vg):
    with pytest.raises(UFLException):
        vf * vg


def test_add_a_v(a, v):
    with pytest.raises(UFLException):
        a + v


def test_add_vf_b(vf, b):
    with pytest.raises(UFLException):
        vf + b


def test_add_vectorexpr_b(vg, v, u, vf, b):
    tmp = vg + v + u + vf
    with pytest.raises(UFLException):
        tmp + b
