
from itertools import izip
from ufl import *
from ufl.classes import *
from ufl.algorithms import *
from ufl.algorithms.transformations import Transformer, transform_integrands

# temporarily importing these to keep other code working
from ufl.algorithms.old_ad import compute_diff, propagate_spatial_derivatives, compute_form_derivative

#TODO:
#- Make basic tests for the below code!
#- Implement forward mode AD.
#- Implement reverse mode AD.
#- Nonscalar expressions have been ignored in the below code!
#- Nonscalar derivatives have been ignored in the below code!

class PartialDerivativeCalculator(Transformer):
    "NB! The main reason for keeping this out of the Expr hierarchy is to avoid user mistakes."
    def __init__(self):
        Transformer.__init__(self)
    
    def expr(self, x):
        ufl_error("No handler for %s" % str(type(x)))
    
    def terminal(self, x):
        ufl_error("No handler for %s" % str(type(x)))
     
    def sum(self, x):
        # TODO: Handle non-scalars
        _1 = IntValue(1)
        return tuple(_1 for o in x.operands())
    
    # FIXME: Implement partial derivatives of all operators

def apply_ad(expr, ad_routine, G=None):
    # Create a fresh DAG for root if none is provided
    if G is None:
        V = []
        E = []
        G = (V, E)
    
    # Record terminals in graph
    if isinstance(expr, Terminal):
        V, E = G
        V.append(expr)
        return expr
    
    # Found a derivative, need to do something more interesting...
    if isinstance(expr, Derivative):
        parent_G = G
        
        # Create a fresh DAG for subtree
        V = []
        E = []
        G = (V, E)
        
        # Handle children first to make sure they have no derivatives themselves
        ops = expr.operands()
        ops2 = [apply_ad(o, ad_routine, G) for o in ops]
        
        # Reuse or reconstruct
        expr2 = expr if all(a is b for (a,b) in izip(ops, ops2)) else expr._uflid(*ops2)
        
        # Compute derivative
        dexpr = ad_routine(expr2, G) #, parent_G)
        
        # TODO: We may be able to augment parent_G cheaper during AD itself
        # Record new subtree (without derivatives!) in parent graph
        expr2 = apply_ad(dexpr, ad_routine, parent_G)
        
    else:
        # Handle children first to make sure they have no derivatives themselves
        ops = expr.operands()
        ops2 = [apply_ad(o, ad_routine, G) for o in ops]
        
        # Reuse or reconstruct
        expr2 = expr if all(a is b for (a,b) in izip(ops, ops2)) else expr._uflid(*ops2)
        
        # Record updated expr2 in parent graph
        V, E = G
        V.append(expr2)
        for o in expr2.operands():
            E.append((expr2, o)) # NB! No indices in use here, storing expressions directly!
    
    return expr2

def reverse_ad(expr, G): # FIXME: Finish this!
    # --- Forward sweep expressions have already been recorded as vertices in the DAG
    V, E = G
    
    # FIXME: Transform graph structure to index based linear graph?
    
    # We want to differentiate f w.r.t. x:
    f, x = expr.operands()
    
    # --- Compute all partial derivatives of all v = V[i] (FIXME!)
    pdc = PartialDerivativeCalculator()
    c = {}
    for i, v in enumerate(V):
        pdiffs = pdc.visit(v)
        vi_edges = FIXME
        for (j, dvidvj) in zip(vi_edges, pdiffs):
            c[(i,j)] = dvidvj
    
    # --- Reverse accumulation (FIXME!)
    
    if isinstance(expr, SpatialDerivative):
        pass
    
    if isinstance(expr, VariableDerivative):
        pass
    
    if isinstance(expr, FunctionDerivative):
        pass
    
    df = FIXME
    
    return df

# TODO: While it works fine to propagate d/dx to terminals in
#     (a + b).dx(i) => (a.dx(i) + b.dx(i))
# this case doesn't work out as easily:
#     f[i].dx(i) => f.dx(i)[i] # rhs here doesn't make sense
# What is the best way to handle this?
# We must keep this in mind during extension
# of AD to tensor expressions!
def expand_derivatives(expr):
    """Expand all derivatives of expr.
    
    NB! This functionality is not finished!
    
    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or FunctionDerivative objects left, and SpatialDerivative
    objects have been propagated to Terminal nodes."""
    def _transform(expr):
        return apply_ad(expr, reverse_ad)
    return transform_integrands(expr, _transform)

if __name__ == "__main__":
    from ufl import *
    e = FiniteElement("CG", triangle, 1)
    v = TestFunction(e)
    f = Function(e)
    a = f*v*dx
    da = expand_derivatives(a)
    print 
    print a
    print
    print da 
    print

