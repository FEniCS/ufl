#!/usr/bin/env py.test
# -*- coding: utf-8 -*-
import pytest

from ufl import *
from ufl.form import BaseForm


@pytest.fixture
def element():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    return element


@pytest.fixture
def mass():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    return u * v * dx


@pytest.fixture
def stiffness():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    return inner(grad(u), grad(v)) * dx


def test_form_replace(mass, stiffness):
    v, u = mass.arguments()

    assert v.number() == 0
    assert u.number() == 1
    assert stiffness.arguments() == (v, u)

    assert (v * dx).arguments() == (v,)
    assert (v * dx + v * ds).arguments() == (v,)
    assert (u * v * dx(1) + v * u * dx(2)).arguments() == (v, u)

    replaced_stiffness = replace(stiffness, {v : u})
    
    assert replaced_stiffness.arguments() == (u,)
