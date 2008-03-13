#!/usr/bin/env python

"""
Mathematical functions.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 13th 2008"

from ufl_io import *
from base import *

### Functions

class MathFunction(UFLObject):
    def __init__(self, name, argument):
        ufl_assert(is_true_scalar(argument), "Need scalar.")
        self.name     = name
        self.argument = argument
    
    def operands(self):
        return (self.argument,)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __repr__(self):
        return "%s(%s)" % (self.name, repr(self.argument))

# functions exposed to the user:

def sqrt(f):
    return MathFunction("sqrt", f)

def exp(f):
    return MathFunction("exp", f)

def ln(f):
    return MathFunction("ln", f)

def cos(f):
    return MathFunction("cos", f)

def sin(f):
    return MathFunction("sin", f)

def floor(f):
    return MathFunction("floor", f)

def ceil(f):
    return MathFunction("ceil", f)

