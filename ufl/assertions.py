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
    condition or error(*message)

def assert_instance(o, c):
    "Assert that the object o is an istance of class c."
    isinstance(o, c) or expecting_instance(o, c)

def assert_expr(o):
    "Assert that the object o is an Expr istance."
    from ufl.expr import Expr
    isinstance(o, Expr) or expecting_expr(o)

def assert_terminal(o):
    "Assert that the object o is a Terminal istance."
    from ufl.terminal import Terminal
    isinstance(o, Terminal) or expecting_terminal(o)

def assert_true_scalar(o):
    "Assert that the object o is a proper scalar expression with no free indices."
    from ufl.scalar import is_true_ufl_scalar
    is_true_ufl_scalar(o) or expecting_true_ufl_scalar(o)

