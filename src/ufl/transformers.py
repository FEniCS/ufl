#!/usr/bin/env python
    
"""
This module contains algorithms based on a visitor-like algorithm design pattern
suited for transforming expression trees from one representation to another.
(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase).
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-14-03"

from all import *


class UFLTransformer:
    """Base class for a visitor-like algorithm design pattern used to 
       transform expression trees from one representation to another."""
    def __init__(self):
        self._f = {}
        self.register(UFLObject, self._default)
    
    def register(self, classobject, function):
        self._f[classobject] = function
    
    def visit(self, o):
        return self._f.get(o.__class__, self._f[UFLObject])(o)
    
    def _default(self, o):
        if isinstance(o, Terminal):
            return o
        operandss = [self.visit(oo) for oo in o.operands()]
        return o.__class__(*operands)


class TreeFlattener(UFLTransformer):
    def __init__(self):
        UFLTransformer.__init__(self)
        self.register(Sum,       self.flatten_sum_or_product)
        self.register(Product,   self.flatten_sum_or_product)
    
    def flatten_sum_or_product(self, o):
        if isinstance(o, Terminal):
            return o
        operands = []
        for a in o.operands():
            b = self.visit(a)
            if isinstance(b, o.__class__):
                operands.extend(b.operands())
            else:
                operands.append(b)
        return o.__class__(*operands)


class IndexEvaluator(UFLTransformer): # FIXME: how to do this?
    def __init__(self):
        UFLTransformer.__init__(self)
        self.register(UFLObject,   self.default)
        
        self.register(Sum,         self.sum)
        self.register(Product,     self.product)
        self.register(Sub,         self.sub)
        self.register(Division,    self.division)
        self.register(Power,       self.power)
        
        self.register(Grad,        self.grad)
        self.register(Div,         self.div)
        self.register(Curl,        self.curl)
        self.register(Wedge,       self.wedge)
    
    def default(self, o):
        operands = []
        idx_list = []
        idx_set  = []
        for a in o.operands():
            b = self.visit(a)
            operands.append(b)
        if isinstance(o, Terminal):
            return o
        return o.__class__(*operands)



class UFL2Something(UFLTransformer):
    """Base class for a visitor-like algorithm design pattern used to 
       transform expression trees from one representation to another."""
    def __init__(self):
        UFLTransformer.__init__(self)
        # default functions for all classes that use built in python operators
        self.register(Sum,         self.sum)
        self.register(Product,     self.product)
        self.register(Sub,         self.sub)
        self.register(Division,    self.division)
        self.register(Power,       self.power)
    
    def sum(self, o):
        return sum(self.visit(i) for i in o.operands())
    
    def product(self, o):
        return product(self.visit(i) for i in o.operands())
    
    def sub(self, o):
        a, b = o.operands()
        a, b = self.visit(a), self.visit(b)
        return a - b
    
    def division(self, o):
        a, b = o.operands()
        a, b = self.visit(a), self.visit(b)
        return a / b
    
    def power(self, o):
        a, b = o.operands()
        a, b = self.visit(a), self.visit(b)
        return a ** b


class UFL2UFL(UFL2Something):
    def __init__(self):
        UFL2Something.__init__(self)
        self.register(UFLObject,   self.default)

        # TODO: compound tensor operations (inner, dot, outer, ...)
        
        # compound differential operators:
        self.register(Grad,        self.grad)
        self.register(Div,         self.div)
        self.register(Curl,        self.curl)
        
        # terminal objects:
        self.register(FacetNormal, self.facet_normal)
        
        # TODO: add operations for all classes here...
    
    def default(self, o):
        if isinstance(o, Terminal):
            return o
        operands = [self.visit(oo) for i in o.operands()]
        return o.__class__(*operands)
    
    def grad(self, o):
        f, = o.operands()
        f = self.visit(f)
        return grad(f)
    
    def div(self, o):
        f, = o.operands()
        f = self.visit(f)
        return div(f)
    
    def curl(self, o):
        f, = o.operands()
        f = self.visit(f)
        return curl(f)
    
    def facet_normal(self, o):
        return FacetNormal()



if __name__ == "__main__":
    a = FiniteElement("CG", "triangle", 1)
    b = VectorElement("CG", "triangle", 1)
    c = TensorElement("CG", "triangle", 1)
    
    u = TrialFunction(a)
    v = TestFunction(a)
    
    g = Function(a, "g")
    c = Constant(a.polygon, "c")
    
    u = TrialFunction(a)
    v = TestFunction(a)
    
    g = Function(a, "g")

    f = u + v + g + 3
    print "f:"
    print f   
    vis = TreeFlattener()
    print "f:"
    print vis.visit(f)
    print ""


