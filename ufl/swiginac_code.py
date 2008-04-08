
# TEMPORARY FILE; REMOVE BEFORE RELEASE (or keep swiginac/ and sympy/ modules with evaluators? probably have to specialize with syfi code anyway)

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

