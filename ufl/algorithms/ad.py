"""Front-end for AD routines."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-12-28 -- 2009-02-18"

from itertools import izip
from ufl.log import debug, error
from ufl.assertions import ufl_assert
from ufl.classes import Terminal, Expr, Derivative, Tuple, SpatialDerivative, VariableDerivative, FunctionDerivative, FiniteElement, TestFunction, Function
#from ufl.algorithms import *
from ufl.algorithms.analysis import extract_classes
from ufl.algorithms.transformations import transform_integrands, expand_compounds, Transformer

from ufl.algorithms.reverse_ad import reverse_ad
from ufl.algorithms.forward_ad import forward_ad

class ADApplyer(Transformer):
    def __init__(self, ad_routine, dim):
        Transformer.__init__(self)
        self._ad_routine = ad_routine
        self._dim = dim
    
    def terminal(self, e):
        return e
    
    def expr(self, e, *ops):
        e = Transformer.reuse_if_possible(self, e, *ops)
        if isinstance(e, Derivative):
            e = self._ad_routine(e, self._dim)
        return e

def expand_derivatives(form):
    """Expand all derivatives of expr.
    
    NB! This functionality is not finished!
    
    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or FunctionDerivative objects left, and SpatialDerivative
    objects have been propagated to Terminal nodes."""
    
    # No derivatives? Then we do nothing. TODO: Can enable this later for efficiency, but omitting it is a good test of the algorithms.
    #if not any(issubclass(c, Derivative) for c in extract_classes(form)):
    #    return form

    # TODO: How to switch between forward and reverse mode? Can we pick the best in each context?
    ad_routine = forward_ad
    #ad_routine = reverse_ad

    cell = form.cell()
    dim = None if cell is None else cell.d

    def _expand_derivatives(expression):
        expression = expand_compounds(expression, dim)
        aa = ADApplyer(ad_routine, dim)
        return aa.visit(expression)

    return transform_integrands(form, _expand_derivatives)

if __name__ == "__main__":
    from ufl import triangle, FiniteElement, TestFunction, Function, expand_derivatives
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


