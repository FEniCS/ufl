#!/usr/bin/env python
    
"""
This module contains algorithms based on a visitor-like algorithm design pattern
suited for transforming expression trees from one representation to another.
(Organizing algorithms by implementation technique is a temporary strategy
only to be used during the current experimental implementation phase).
"""

__version__ = "0.1"
__authors__ = "Martin Sandve Alnes"
__copyright__ = __authors__ + " (2008)"
__licence__ = "GPL" # TODO: which licence?
__date__ = "17th of December 2008"

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
        return o.fromoperands(self.visit(oo) for oo in o.operands())


class TreeFlattener(UFLTransformer):
    def __init__(self):
        UFLTransformer.__init__(self)
        self.register(Sum,       self.flatten_sum_or_product)
        self.register(Product,   self.flatten_sum_or_product)
    
    def flatten_sum_or_product(self, o):
        operands = []
        for a in o.operands():
            b = self.visit(a)
            if isinstance(b, o.__class__):
                operands.extend(b.operands())
            else:
                operands.append(b)
        return o.fromoperands(operands)


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
        return o.fromoperands(operands)



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
        self.register(Wedge,       self.wedge)
        
        # terminal objects:
        self.register(FacetNormal, self.facet_normal)
        
        # TODO: add operations for all classes here...
    
    def default(self, o):
        return o.fromoperands(self.visit(oo) for i in o.operands())
    
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
    
    def wedge(self, o):
        a, b = o.operands()
        a = self.visit(a)
        b = self.visit(b)
        return wedge(a, b)
    
    def facet_normal(self, o):
        return FacetNormal()



# TODO: move this to SyFi code, finish and apply, then add quadrature support
from swiginac import *
#from sfc.symbolic_utils import *
class SwiginacEvaluator(UFL2Something):
    def __init__(self):
        UFL2Something.__init__(self)
        
        self.register(Grad,          self.grad)
        self.register(Div,           self.div)
        self.register(Curl,          self.curl)
        self.register(BasisFunction, self.basis_function)
        
        self.register(FacetNormal, self.facet_normal)
        
        # TODO: add a whole lot of other operations here...
        # TODO: take some context information in constructor, add sfc.Integral object self.itg, perhaps name this class "IntegralBuilder"?
        self.itg = None # built from context
        self.sfc = None # sfc.symbolic_utils
        
        self.reset()
    
    def reset(self):
        self.tokens = []
    
    def basis_function(self, o):
        i = o.index # TODO: fix this
        return self._basis_function[i]
    
    def grad(self, o): # TODO: must support token "barriers", build on general derivative code
        f, = o.operands()
        f = self.visit(f)
        GinvT = self.itg.GinvT()
        return self.sfc.grad(f, GinvT)
    
    def div(self, o): # TODO: must support token "barriers", build on general derivative code
        f, = o.operands()
        f = self.visit(f)
        GinvT = self.itg.GinvT()
        return self.sfc.div(f, GinvT)
    
    def curl(self, o): # TODO: must support token "barriers", build on general derivative code
        f, = o.operands()
        f = self.visit(f)
        GinvT = self.itg.GinvT()
        return self.sfc.curl(f, GinvT)
    
    def facet_normal(self, o):
        return symbol("n")
        #return self.itg.n()


if __name__ == "__main__":
    a = FiniteElement("CG", "triangle", 1)
    b = VectorElement("CG", "triangle", 1)
    c = TensorElement("CG", "triangle", 1)
    
    u = TrialFunction(a)
    v = TestFunction(a)
    
    g = Function(a, "g")
    c = Constant(a.polygon, "c")
    
    e = SwiginacEvaluator()
    u = FacetNormal()
    n = e.visit(u)
    print n, type(n)


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

