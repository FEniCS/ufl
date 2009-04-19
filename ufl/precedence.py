"Precedence handling."

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-27 -- 2009-04-19"

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import product, mergedicts, subdict
from ufl.expr import Expr
from ufl.terminal import Terminal

def parstr(child, parent, pre="(", post=")"):
    s = str(child)
    # We want child to be evaluated fully first,
    # so if the parent has higher precedence
    # we wrap in ().

    # Operators where operands are always parenthesized 
    if parent._precedence == 0:
        return pre + s + post

    # If parent operator binds stronger than child, must parenthesize child
    if parent._precedence > child._precedence:
        return pre + s + post

    # Nothing needed
    return s
