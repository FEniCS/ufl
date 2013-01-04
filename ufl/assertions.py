"""This module provides assertion functions used by the UFL implementation."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
#
# First added:  2009-01-28
# Last changed: 2011-06-02

from ufl.log import error

#--- Standardized error messages ---

expecting_instance        = lambda v, c: error("Expecting %s instance, not %s." % (c.__name__, repr(v)))
expecting_python_scalar   = lambda v:    error("Expecting Python scalar, not %s." % repr(v))
expecting_expr            = lambda v:    error("Expecting Expr instance, not %s." % repr(v))
expecting_terminal        = lambda v:    error("Expecting Terminal instance, not %s." % repr(v))
expecting_true_ufl_scalar = lambda v:    error("Expecting UFL scalar expression with no free indices, not %s." % repr(v))

#--- Standardized assertions ---

def ufl_assert(condition, *message):
    "Assert that condition is true and otherwise issue an error with given message."
    if not condition: error(*message)

