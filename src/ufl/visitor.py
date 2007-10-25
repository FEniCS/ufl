
from ufl import *

count = 0

class UFLVisitor:
    def __init__(self):
        self._f = {}
        self.register(UFLObject, self._uflobject)
    
    def _uflobject(self, o):
        for i in o.ops():
            self.visit(i)
    
    def register(self, classobject, function):
        self._f[classobject] = function
    
    def visit(self, o):
        f = self._f.get(o.__class__, self._f[UFLObject])
        f(o)


class MyVisitor(UFLVisitor):
    def __init__(self):
        UFLVisitor.__init__(self)
        self.register(Sum, self.sum)
        self.register(Product, self.product)
    
    def sum(self, o):
        print o
        self._uflobject(o)
    
    def product(self, o):
        print o
        self._uflobject(o)


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


if __name__ == "__main__":
    a = FiniteElement("La", "tr", 1)
    b = VectorElement("La", "tr", 1)
    c = TensorElement("La", "tr", 1)
    print a, b, c
    
    u = TrialFunction(a)
    v = TestFunction(a)
    print u, v
    
    g = Function(a, "g")
    c = Constant(a.polygon, "c")
    
    f = v * 3 + u + v * 3 / g - exp(c**g)
    
    vis = MyVisitor()
    print vis.visit(f)
    
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
    
