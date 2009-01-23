"""This module defines classes for conditional expressions."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-20 -- 2009-01-09"

from ufl.log import ufl_assert, warning, error
from ufl.expr import Expr
from ufl.scalar import as_ufl

#--- Condition classes --- 

class Condition(Expr):
    def __init__(self, name, left, right):
        Expr.__init__(self)
        self._name = name
        self._left = as_ufl(left)
        self._right = as_ufl(right)
        ufl_assert(self._left.shape() == () \
            and  self._right.shape() == (),
            "Expecting scalar arguments.")
        ufl_assert(self._left.free_indices() == () \
            and self._right.free_indices() == (),
            "Expecting scalar arguments.")
        
    def operands(self):
        # Condition should never be constructed directly,
        # so these two arguments correspond to the constructor
        # arguments of the subclasses EQ etc.
        return (self._left, self._right)

    def free_indices(self):
        error("Calling free_indices on Condition is an error.")
    
    def index_dimensions(self):
        error("Calling index_dimensions on Condition is an error.")

    def shape(self):
        error("Calling shape on Condition is an error.")
    
    def __str__(self):
        return "(%s) %s (%s)" % (self._left, self._name, self._right)

class EQ(Condition):
    def __init__(self, left, right):
        Condition.__init__(self, "==", left, right)
    
    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return a == b

    def __repr__(self):
        return "EQ(%r, %r)" % (self._left, self._right)

class NE(Condition):
    def __init__(self, left, right):
        Condition.__init__(self, "!=", left, right)
    
    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return a != b
    
    def __repr__(self):
        return "NE(%r, %r)" % (self._left, self._right)

class LE(Condition):
    def __init__(self, left, right):
        Condition.__init__(self, "<=", left, right)
    
    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return a <= b
    
    def __repr__(self):
        return "LE(%r, %r)" % (self._left, self._right)

class GE(Condition):
    def __init__(self, left, right):
        Condition.__init__(self, ">=", left, right)
    
    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return a >= b
    
    def __repr__(self):
        return "GE(%r, %r)" % (self._left, self._right)

class LT(Condition):
    def __init__(self, left, right):
        Condition.__init__(self, "<", left, right)
    
    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return a < b
    
    def __repr__(self):
        return "LT(%r, %r)" % (self._left, self._right)

class GT(Condition):
    def __init__(self, left, right):
        Condition.__init__(self, ">", left, right)
    
    def evaluate(self, x, mapping, component, index_values):
        a = self._left.evaluate(x, mapping, component, index_values)
        b = self._right.evaluate(x, mapping, component, index_values)
        return a > b
    
    def __repr__(self):
        return "GT(%r, %r)" % (self._left, self._right)

#--- Conditional expression (condition ? true_value : false_value) ---

class Conditional(Expr):
    __slots__ = ("_condition", "_true_value", "_false_value")
    
    def __init__(self, condition, true_value, false_value):
        Expr.__init__(self)
        ufl_assert(isinstance(condition, Condition), "Expectiong condition as first argument.")
        true_value = as_ufl(true_value)
        false_value = as_ufl(false_value)
        tsh = true_value.shape()
        fsh = false_value.shape()
        ufl_assert(tsh == fsh, "Shape mismatch between conditional branches.")
        tfi = true_value.free_indices()
        ffi = false_value.free_indices()
        ufl_assert(tfi == ffi, "Free index mismatch between conditional branches.")
        self._condition = condition
        self._true_value = true_value
        self._false_value = false_value

    def operands(self):
        return (self._condition, self._true_value, self._false_value)

    def free_indices(self):
        return self._true_value.free_indices()

    def index_dimensions(self):
        return self._true_value.index_dimensions()

    def shape(self):
        return self._true_value.shape()
    
    def evaluate(self, x, mapping, component, index_values):
        c = self._condition.evaluate(x, mapping, component, index_values)
        if c:
            a = self._true_value
        else:
            a = self._false_value
        return a.evaluate(x, mapping, component, index_values)

    def __str__(self):
        return "(%s) ? (%s) : (%s)" % self.operands()
    
    def __repr__(self):
        return "Conditional(%r, %r, %r)" % self.operands()
    
