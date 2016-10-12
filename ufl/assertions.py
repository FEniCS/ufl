# -*- coding: utf-8 -*-
"""This module provides assertion functions used by the UFL implementation."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.

from ufl.log import error


# TODO: Move to this file and make other files import from here
from ufl.core.expr import ufl_err_str


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
