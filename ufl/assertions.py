# -*- coding: utf-8 -*-
"""This module provides assertion functions used by the UFL implementation."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl_legacy.log import error


# TODO: Move to this file and make other files import from here
from ufl_legacy.core.expr import ufl_err_str


# TODO: Use these and add more
# --- Standardized error messages ---

def expecting_instance(v, c):
    error("Expecting %s instance, not %s." % (c.__name__, ufl_err_str(v)))


def expecting_python_scalar(v):
    error("Expecting Python scalar, not %s." % ufl_err_str(v))


def expecting_expr(v):
    error("Expecting Expr instance, not %s." % ufl_err_str(v))


def expecting_terminal(v):
    error("Expecting Terminal instance, not %s." % ufl_err_str(v))


def expecting_true_ufl_scalar(v):
    error("Expecting UFL scalar expression with no free indices, not %s." % ufl_err_str(v))


# --- Standardized assertions ---

# TODO: Stop using this
def ufl_assert(condition, *message):
    "Assert that condition is true and otherwise issue an error with given message."
    if not condition:
        error(*message)
