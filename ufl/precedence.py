"Precedence handling."

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-03-27 -- 2009-03-27"

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import product, mergedicts, subdict
from ufl.expr import Expr
from ufl.terminal import Terminal

# TODO: Move precedence lists here from ufl2latex

def parstr(child, parent, pre="(", post=")"):
    s = str(child)
    if not isinstance(child, Terminal): # TODO: Replace by precedence check
        return pre + s + post
    return s

