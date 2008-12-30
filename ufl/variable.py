"""Defines symbol and variable constructs."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-20 -- 2008-12-22"

from ufl.common import Counted
from ufl.output import ufl_assert
from ufl.expr import Expr
from ufl.terminal import Terminal

class Variable(Terminal, Counted):
    """A Variable is a representative for another expression.
    
    It will be used by the end-user mainly for:
    - Defining a quantity to differentiate w.r.t. using diff.
    
    Internally, it is also used for:
    - Marking good spots to split an expression for optimized computation.
    - Reuse of expressions during e.g. automatic differentation.
    """
    __slots__ = ("_expression",)# "_diffcache")
    _globalcount = 0
    def __init__(self, expression, count=None):
        Terminal.__init__(self)
        Counted.__init__(self, count)
        ufl_assert(isinstance(expression, Expr), "Expecting an Expr.")
        self._expression = expression
        #self._diffcache = defaultdict(list) # FIXME: This needs better definition...
    
    def operands(self):
        return ()
    
    def free_indices(self):
        return self._expression.free_indices()
    
    def index_dimensions(self):
        return self._expression.index_dimensions()
    
    def shape(self):
        return self._expression.shape()
    
    def cell(self):
        return self._expression.cell()
    
    def __eq__(self, other):
        return isinstance(other, Variable) and self._count == other._count
        
    def __str__(self):
        return "Variable(%s, %d)" % (self._expression, self._count)
    
    def __repr__(self):
        return "Variable(%r, %r)" % (self._expression, self._count)
