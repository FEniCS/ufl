"""This module provides assertion functions used by the UFL implementation."""

__author__ = "Martin Sandve Alnes"
__date__ = "2009-01-28 -- 2009-01-28"

from ufl.log import error

def ufl_assert(condition, *message):
    "Assert that condition is true and otherwise issue an error with given message."
    if not condition:
        error(*message)

def assert_instance(o, c):
    "Assert that the object o is an istance of class c."
    if not isinstance(o, c):
        error("Expecting %s instance, not %s." % (c.__name__, str(o)))

def assert_expr(o):
    "Assert that the object o is an Expr istance."
    from ufl.expr import Expr
    if not isinstance(o, Expr):
        error("Expecting Expr instance, not %s." % str(o))

def assert_terminal(o):
    "Assert that the object o is a Terminal istance."
    from ufl.terminal import Terminal
    if not isinstance(o, Terminal):
        error("Expecting Terminal instance, not %s." % str(o))

def assert_true_scalar(o):
    "Assert that the object o is a proper scalar expression with no free indices."
    from ufl.scalar import is_true_ufl_scalar
    if not is_true_ufl_scalar(o):
        error("Expecting true scalar expression, not %s." % str(o))

