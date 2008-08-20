"""This module defines classes for conditional expressions."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-20 -- 2008-08-20"

from .output import ufl_assert
from .base import UFLObject, Terminal
from .indexing import compare_shapes


#--- Condition classes --- 

class Condition(Terminal):
    def __init__(self, name, left, right):
        self._name = name
        self._left = left
        self._right = right

class EQ(Condition):
    def __init__(self, left, right):
        Condition.__init__("==", left, right)

class NE(Condition):
    def __init__(self, left, right):
        Condition.__init__("!=", left, right)

class LE(Condition):
    def __init__(self, left, right):
        Condition.__init__("<=", left, right)

class GE(Condition):
    def __init__(self, left, right):
        Condition.__init__(">=", left, right)

class LT(Condition):
    def __init__(self, left, right):
        Condition.__init__("<", left, right)

class GT(Condition):
    def __init__(self, left, right):
        Condition.__init__(">", left, right)


#--- Conditional expression (condition ? true_value : false_value) ---

class Conditional(UFLObject):
    def __init__(self, condition, true_value, false_value):
        if is_python_scalar(true_value):
            true_value = Number(true_value)
        if is_python_scalar(false_value):
            false_value = Number(false_value)
        ufl_assert(isinstance(condition, Condition), "Expectiong condition as first argument.")
        ufl_assert(isinstance(true_value, UFLObject), "Expectiong UFL expression as second argument.")
        ufl_assert(isinstance(false_value, UFLObject), "Expectiong UFL expression as third argument.")
        tsh = true_value.shape()
        fsh = false_value.shape()
        ufl_assert(compare_shapes(tsh, fsh), "Shape mismatch between conditional branches.")
        tfi = true_value.free_indices()
        ffi = false_value.free_indices()
        ufl_assert(tfi == ffi, "Free index mismatch between conditional branches.")
        self._condition = condition
        self._true_value = true_value
        self._false_value = false_value
        self._shape = tsh
        self._free_indices = tfi

    def operands(self):
        return (self._condition, self._true_value, self._false_value)

    def free_indices(self):
        return self._free_indices

    def shape(self):
        return self._shape

    def __str__(self):
        return "(%s) ? (%s) : (%s)" % self.operands()
    
    def __repr__(self):
        return "Conditional(%r, %r, %r)" % self.operands()
    