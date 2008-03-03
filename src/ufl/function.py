#!/usr/bin/env python

"""
Mathematical functions.
"""

from base import *

### Functions

class UFLFunction(UFLObject):
    def __init__(self, name, argument):
        self.name     = name
        self.argument = argument
    
    def ops(self):
        return (self.argument,)
    
    def __repr__(self):
        return "%s(%s)" % (self.name, repr(self.argument))

# functions exposed to the user:

def sqrt(f):
    return UFLFunction("sqrt", f)

def exp(f):
    return UFLFunction("exp", f)

def ln(f):
    return UFLFunction("ln", f)

def cos(f):
    return UFLFunction("cos", f)

def sin(f):
    return UFLFunction("sin", f)

