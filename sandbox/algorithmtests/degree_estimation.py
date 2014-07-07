

from ufl import *
from ufl.common import *
from ufl.classes import *
from ufl.algorithms import *
from ufl.algorithms.transformations import Transformer, apply_transformer

class DegreeEstimator(Transformer):
    def __init__(self):
        Transformer.__init__(self)
    
    def terminal(self, v):
        return 0
    
    def expr(self, v, *ops):
        return max(ops)
    
    def form_argument(self, v):
        return v.element().degree()
    
    def spatial_derivative(self, v, f, i):
        return max(f-1, 0)
    
    def product(self, v, *ops):
        return sum(ops)
    
    def power(self, v, a, b):
        f, g = v.operands()
        try:
            gi = int(g)
            return a*gi
        except:
            pass
        # Something to a non-integer power... TODO: How to handle this?
        if b > 0:
            return a*b
        return a

def estimate_max_quad_degree(e):
    return apply_transformer(e, DegreeEstimator())

from ufl.testobjects import *
if __name__ == "__main__":
    print((estimate_max_quad_degree(v)))
    print((estimate_max_quad_degree(u*v)))
    print((estimate_max_quad_degree(vv[0])))
    print((estimate_max_quad_degree(u*vv[0])))
    print((estimate_max_quad_degree(vu[0]*vv[0])))
    print((estimate_max_quad_degree(u*v)))
    print((estimate_max_quad_degree(vu[i]*vv[i])))
    print()
    V1 = FiniteElement("CG", triangle, 1)
    V2 = FiniteElement("CG", triangle, 2)
    VM = V1 + V2
    v = TestFunction(V1)
    u = TrialFunction(V2)
    f, g = Functions(VM)
    
    print((estimate_max_quad_degree(u)))
    print((estimate_max_quad_degree(v)))
    print((estimate_max_quad_degree(f)))
    print((estimate_max_quad_degree(g)))
    print((estimate_max_quad_degree(u*v)))
    print((estimate_max_quad_degree(f*v)))
    print((estimate_max_quad_degree(g*v)))
    print((estimate_max_quad_degree(g*u*v)))
    print((estimate_max_quad_degree(f*g*u.dx(0)*v.dx(0))))
    print((estimate_max_quad_degree(g**3*v + f*v)))

