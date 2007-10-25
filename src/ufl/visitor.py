
from ufl import *


class UFLVisitor:
    def __init__(self):
        self._f = {}
        self.register(UFLObject, self._default)
    
    def register(self, classobject, function):
        self._f[classobject] = function
    
    def visit(self, o):
        return self._f.get(o.__class__, self._f[UFLObject])(o)
    
    def _default(self, o):
        for i in o.ops():
            self.visit(i)


class BasisFunctionFinder(UFLVisitor):
    def __init__(self):
        UFLVisitor.__init__(self)
        self.register(TestFunction,  self.test_function)
        self.register(TrialFunction, self.trial_function)
        self.reset()
    
    def reset(self):
        self.testfunctions = set()
        self.trialfunctions = set()
    
    def test_function(self, o):
        self.testfunctions.add(o)
    
    def trial_function(self, o):
        self.trialfunctions.add(o)


class CoefficientFinder(UFLVisitor):
    def __init__(self):
        UFLVisitor.__init__(self)
        self.register(Function, self.function)
        self.register(Constant, self.constant)
        self.register(UFLCoefficient, self.unknown_coefficient_type)
        self.reset()
    
    def reset(self):
        self.functions = set()
        self.constants = set()
        self.unknown_coefficients = set()
    
    def function(self, o):
        self.functions.add(o)
    
    def constant(self, o):
        self.constants.add(o)
    
    def unknown_coefficient_type(self, o):
        self.unknown_coefficients.add(o)


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
        elif not isinstance(ro, (Number, Symbol, Variable)):
            self.handled.add(ro)
            for i in o.ops():
                self.visit(i)


class UFLTransformer:
    def __init__(self):
        self._f = {}
        self.register(UFLObject, self._default)
    
    def register(self, classobject, function):
        self._f[classobject] = function
    
    def visit(self, o):
        return self._f.get(o.__class__, self._f[UFLObject])(o)
    
    def _default(self, o):
        return o.fromops(self.visit(i) for i in o.ops())


class TreeFlattener(UFLTransformer):
    def __init__(self):
        UFLTransformer.__init__(self)
        self.register(Sum,       self.flatten_sum_or_product)
        self.register(Product,   self.flatten_sum_or_product)
    
    def flatten_sum_or_product(self, o):
        ops = []
        for i in o.ops():
            j = self.visit(i)
            if isinstance(j, o.__class__):
                ops.extend(j.ops())
            else:
                ops.append(j)
        return o.fromops(ops)


#from swiginac import *
#from sfc.symbolic_utils import *
class SwiginacEvaluator(UFLTransformer):
    def __init__(self):
        UFLTransformer.__init__(self)
        
        self.register(Sum,         self.sum)
        self.register(Product,     self.product)
        self.register(Sub,         self.sub)
        self.register(Division,    self.division)
        self.register(Power,       self.power)
        
        self.register(Grad,        self.grad)
        self.register(Div,         self.div)
        self.register(Curl,        self.curl)
        self.register(Wedge,       self.wedge)
        
        self.register(FacetNormal, self.facet_normal)
        
        # TODO: add a whole lot of other operations here...
        # TODO: take some context information in constructor, add sfc.Integral object self.itg, perhaps name this class "IntegralBuilder"?
        self.itg = None # built from context
        self.sfc = None # sfc.symbolic_utils
        
        self.reset()
    
    def reset(self):
        self.tokens = []
    
    def sum(self, o):
        return sum(self.visit(i) for i in o.ops())
    
    def product(self, o):
        return product(self.visit(i) for i in o.ops())
    
    def sub(self, o):
        a, b = o.ops()
        a, b = self.visit(a), self.visit(b)
        return a - b
    
    def division(self, o):
        a, b = o.ops()
        a, b = self.visit(a), self.visit(b)
        return a / b
    
    def power(self, o):
        a, b = o.ops()
        a, b = self.visit(a), self.visit(b)
        return a ** b
    
    def grad(self, o):
        f, = o.ops()
        f = self.visit(f)
        GinvT = self.itg.GinvT()
        return self.sfc.grad(f, GinvT)
    
    def div(self, o):
        f, = o.ops()
        f = self.visit(f)
        GinvT = self.itg.GinvT()
        return self.sfc.div(f, GinvT)
    
    def curl(self, o):
        f, = o.ops()
        f = self.visit(f)
        GinvT = self.itg.GinvT()
        return self.sfc.curl(f, GinvT)
    
    def wedge(self, o):
        f, = o.ops()
        f = self.visit(f)
        GinvT = self.itg.GinvT()
        return self.sfc.wedge(f, GinvT)
    
    def facet_normal(self, o):
        return self.itg.n()


if __name__ == "__main__":
    a = FiniteElement("La", "tr", 1)
    b = VectorElement("La", "tr", 1)
    c = TensorElement("La", "tr", 1)
    
    u = TrialFunction(a)
    v = TestFunction(a)
    
    g = Function(a, "g")
    c = Constant(a.polygon, "c")
    
    f = v * 3 + u + v * 3 / g - exp(c**g)
    
    vis = BasisFunctionFinder()
    print vis.visit(f)
    print vis.testfunctions
    print vis.trialfunctions
    
    vis = CoefficientFinder()
    print vis.visit(f)
    print vis.functions
    print vis.constants
    print vis.unknown_coefficients

    vis = SubtreeFinder()
    print vis.visit(f)
    print vis.duplicated

    f = u + 1 + v + 2 + g + 3
    print ""
    print f   
    vis = TreeFlattener()
    print ""
    print vis.visit(f)
    print ""

