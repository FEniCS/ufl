"""This module provides assertion functions used by the UFL implementation."""

__author__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-02-04"

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

