#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

# This file must be separate from the other arithmetic
# tests to test the effect of this future statment
from __future__ import division

import pytest

from ufl import *
from ufl.classes import Division, FloatValue, IntValue


def test_future_true_float_division(self):
    d = as_ufl(20.0) / 10.0
    self.assertIsInstance(d, FloatValue)
    assert float(d) == 2


def test_future_true_int_division(self):
    # UFL treats all divisions as true division
    d = as_ufl(40) / 7
    self.assertIsInstance(d, FloatValue)
    assert float(d) == 40.0 / 7.0
    #self.assertAlmostEqual(float(d), 40 / 7.0, 15)


def test_future_floor_division_fails(self):
    f = as_ufl(2.0)
    r = as_ufl(4)
    s = as_ufl(5)
    self.assertRaises(NotImplementedError, lambda: r // 4)
    self.assertRaises(NotImplementedError, lambda: r // s)
    self.assertRaises(NotImplementedError, lambda: f // s)
