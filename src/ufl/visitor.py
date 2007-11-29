
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
        self.register(TestFunction,  self.test_function)  # FIXME: just use BasisFunction
        self.register(TrialFunction, self.trial_function)
        self.register(BasisFunction, self.basis_function)
        self.reset()
    
    def reset(self):
        self.testfunctions  = set()
        self.trialfunctions = set()
        self.basisfunctions = set()
    
    def test_function(self, o):
        self.testfunctions.add(o)
        self.basisfunctions.add(o)
    
    def trial_function(self, o):
        self.trialfunctions.add(o)
        self.basisfunctions.add(o)
    
    def basis_function(self, o):
        self.basisfunctions.add(o)


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
        return o.fromops(self.visit(oo) for oo in o.ops())


class TreeFlattener(UFLTransformer):
    def __init__(self):
        UFLTransformer.__init__(self)
        self.register(Sum,       self.flatten_sum_or_product)
        self.register(Product,   self.flatten_sum_or_product)
    
    def flatten_sum_or_product(self, o):
        ops = []
        for a in o.ops():
            b = self.visit(a)
            if isinstance(b, o.__class__):
                ops.extend(b.ops())
            else:
                ops.append(b)
        return o.fromops(ops)


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
        ops = []
        idx_list = []
        idx_set  = []
        for a in o.ops():
            b = self.visit(a)
            ops.append(b)
        return o.fromops(ops)



class UFL2Something(UFLTransformer):
    def __init__(self):
        UFLTransformer.__init__(self)
        # default functions for all classes that use built in python operators
        self.register(Sum,         self.sum)
        self.register(Product,     self.product)
        self.register(Sub,         self.sub)
        self.register(Division,    self.division)
        self.register(Power,       self.power)
    
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
        return o.fromops(self.visit(oo) for i in o.ops())
    
    def grad(self, o):
        f, = o.ops()
        f = self.visit(f)
        return grad(f)
    
    def div(self, o):
        f, = o.ops()
        f = self.visit(f)
        return div(f)
    
    def curl(self, o):
        f, = o.ops()
        f = self.visit(f)
        return curl(f)
    
    def wedge(self, o):
        a, b = o.ops()
        a = self.visit(a)
        b = self.visit(b)
        return wedge(a, b)
    
    def facet_normal(self, o):
        return FacetNormal()



# TODO: move this to SyFi code, finish and apply, then add quadrature support
#from swiginac import *
#from sfc.symbolic_utils import *
class SwiginacEvaluator(UFL2Something):
    def __init__(self):
        UFL2Something.__init__(self)
        
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


# Simpler user interfaces to utilities:

def basisfunctions(u):
    vis = BasisFunctionFinder()
    vis.visit(u)
    # TODO: sort
    return vis.basisfunctions

def coefficients(u):
    vis = CoefficientFinder()
    vis.visit(u)
    # TODO: sort
    return vis.coefficients

def duplications(u):
    vis = SubtreeFinder()
    vis.visit(f)
    return vis.duplicated

def flatten(u):
    vis = TreeFlattener()
    return vis.visit(f)


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

