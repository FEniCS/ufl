"""Mathematical functions."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"


from output import *
from base import *


#--- Function representations ---

class MathFunction(UFLObject):
    def __init__(self, name, argument):
        ufl_assert(is_true_scalar(argument), "Need scalar.")
        self.name     = name
        self.argument = argument
    
    def operands(self):
        return (self.argument,)
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "%s(%s)" % (self.name, str(self.argument))
    
    def __repr__(self):
        return "%s(%s)" % (self.name, repr(self.argument))


#--- Functions exposed to the user ---

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

