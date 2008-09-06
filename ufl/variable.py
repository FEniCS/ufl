"""Defines symbol and variable constructs."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-20 -- 2008-08-18"

from .base import Terminal #UFLObject
from .common import Counted

class Variable(Terminal, Counted): #UFLObject
    """A Variable is a representative for another expression.
    
    It will be used for a few different things:
    - To define a quantity to differentiate w.r.t. using diff.
    - To manually identify good spots to split an expression for optimized computation.
    - Internal use in library algorithms during e.g. automatic differentation.
    """
    __slots__ = ("_expression",)
    _globalcount = 0
    def __init__(self, expression, count=None):
        Counted.__init__(self, count)
        self._expression = expression
    
    def operands(self):
        return ()#self._expression,)
    
    def free_indices(self):
        return self._expression.free_indices()
    
    def shape(self):
        return self._expression.shape()
    
    def __str__(self):
        return "Variable(%s)" % self._expression
    
    def __repr__(self):
        return "Variable(%r, %r)" % (self._expression, self._count)

