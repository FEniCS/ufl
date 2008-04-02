#!/usr/bin/env python

"""
This module contains algorithms based on the visitor pattern, which is mostly
suited to traverse en expression tree and build up some information about the tree
without modifying it.

(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase).
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-04-01"

from all import *


class UFLVisitor:
    def __init__(self):
        self._f = {}
        self.register(UFLObject, self._default)
    
    def register(self, classobject, function):
        self._f[classobject] = function
    
    def visit(self, o):
        return self._f.get(o.__class__, self._f[UFLObject])(o)
    
    def _default(self, o):
        for i in o.operands():
            self.visit(i)


class BasisFunctionFinder(UFLVisitor):
    def __init__(self):
        UFLVisitor.__init__(self)
        self.register(BasisFunction, self.basis_function)
        self.reset()
    
    def reset(self):
        self.basisfunctions = set()
    
    def basis_function(self, o):
        self.basisfunctions.add(o)


class FunctionFinder(UFLVisitor):
    def __init__(self):
        UFLVisitor.__init__(self)
        self.register(Function, self.function)
        self.register(Constant, self.constant)
        self.reset()
    
    def reset(self):
        self.functions = set()
        self.constants = set()
        self.unknown_coefficients = set()
    
    def function(self, o):
        self.functions.add(o)
    
    def constant(self, o):
        self.constants.add(o)


class SubtreeFinder(UFLVisitor):
    def __init__(self):
        UFLVisitor.__init__(self)
        self.register(UFLObject, self.notify)
        self.handled = set()
        self.duplicated = set()
    
    def notify(self, o):
        ro = repr(o)
        if ro in self.handled:
            self.duplicated.add(ro)
        elif not isinstance(ro, (Number, Symbol)):#, Variable)):
            self.handled.add(ro)
            for i in o.operands():
                self.visit(i)
                
    def _notify(self, o):
        ro = repr(o)
        if ro in self.handled:
            self.duplicated.add(ro)
        elif not isinstance(ro, (Number, Symbol)):#, Variable)):
            self.handled.add(ro)
            for i in o.operands():
                self.visit(i)




if __name__ == "__main__":
    a = FiniteElement("Lagrange", "triangle", 1)
    b = VectorElement("Lagrange", "triangle", 1)
    c = TensorElement("Lagrange", "triangle", 1)
    
    u = TrialFunction(a)
    v = TestFunction(a)
    
    g = Function(a, "g")
    c = Constant(a.polygon, "c")
    
    f = v * 3 + u + v * 3 / g - exp(c**g)
    
    vis = BasisFunctionFinder()
    print vis.visit(f)
    
    vis = FunctionFinder()
    print vis.visit(f)
    print vis.functions
    print vis.constants
    print vis.unknown_coefficients

    vis = SubtreeFinder()
    print vis.visit(f)
    print vis.duplicated

