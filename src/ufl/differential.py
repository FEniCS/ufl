#!/usr/bin/env python

"""
Differential operators. Needs work!
"""

from base import *


# objects representing the differential operations:

class DifferentialOperator(UFLObject):
    """For the moment this is just a dummy class to enable "isinstance(o, DifferentialOperator)"."""
    def __init__(self):
        pass

class DiffOperator(DifferentialOperator):
    def __init__(self, x):
        if isinstance(x, int):
            x = p[x]
        elif not isinstance(x, Symbol):
            raise ValueError("x must be a Symbol")
        self.x = x
    
    def __mul__(self, o):
        return diff(o, self.x)

    def __repr__(self):
        return "DiffOperator(%s)" % repr(self.x)

class Diff(DifferentialOperator):
    """The derivative of f with respect to x."""
    def __init__(self, f, x):
        self.f = f
        self.x = x
    
    def operands(self):
        return (self.f, self.x)
    
    def __repr__(self):
        return "Diff(%s, %s)" % (repr(self.f), repr(self.x))

class Grad(DifferentialOperator):
    def __init__(self, f):
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def __repr__(self):
        return "Grad(%s)" % repr(self.f)

class Div(DifferentialOperator):
    def __init__(self, f):
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def __repr__(self):
        return "Div(%s)" % repr(self.f)

class Curl(DifferentialOperator):
    def __init__(self, f):
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def __repr__(self):
        return "Curl(%s)" % repr(self.f)


# functions exposed to the user:

def diff(f, x):
    return Diff(f, x)

def Dx(f, i):
    return Diff(f, x[i]) # TODO: define x[] as symbols somewhere?

def grad(f):
    return Grad(f)

def div(f):
    return Div(f)

def curl(f):
    return Curl(f)

# TODO: What about time derivatives? Can we do something there?
#def Dt(f):
#    return Diff(f, t)

