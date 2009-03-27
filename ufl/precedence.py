"Precedence handling."

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-27 -- 2009-03-27"

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
    #if parent._precedence > child._precedence: # FIXME: Need proper precedence map, and perhaps >=
    if not isinstance(child, Terminal):
        return pre + s + post
    return s
