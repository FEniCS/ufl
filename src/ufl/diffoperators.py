#!/usr/bin/env python

"""
Differential operators. Needs work!
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-14-03"

from output import *
from base import *


# objects representing the differential operations:

class DifferentialOperator(UFLObject):
    """For the moment this is just a dummy class to enable "isinstance(o, DifferentialOperator)"."""
    def __init__(self):
        pass


class Grad(DifferentialOperator):
    def __init__(self, f):
        self.f = f
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking gradient of an expression with free indices, should this be a valid expression? Please provide examples!")
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return self.f.rank() + 1

    def __repr__(self):
        return "Grad(%s)" % repr(self.f)


class Div(DifferentialOperator):
    def __init__(self, f):
        ufl_assert(f.rank() >= 1, "Can't take the divergence of a scalar.")
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking divergence of an expression with free indices, should this be a valid expression? Please provide examples!")
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return self.f.rank() - 1

    def __repr__(self):
        return "Div(%s)" % repr(self.f)


class Curl(DifferentialOperator):
    def __init__(self, f):
        ufl_assert(f.rank()== 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return 1
    
    def __repr__(self):
        return "Curl(%s)" % repr(self.f)


class Rot(DifferentialOperator):
    def __init__(self, f):
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, "FIXME: Taking rot of an expression with free indices, should this be a valid expression? Please provide examples!")
        self.f = f
    
    def operands(self):
        return (self.f, )
    
    def free_indices(self):
        return self.f.free_indices()
    
    def rank(self):
        return 0
    
    def __repr__(self):
        return "Rot(%s)" % repr(self.f)


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

def rot(f):
    return Rot(f)

# TODO: What about time derivatives? Can we do something there?
#def Dt(f):
#    return Diff(f, t)

