"Algorithms related to restrictions."

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-05-14 -- 2009-06-08"

from ufl.expr import Expr
from ufl.assertions import ufl_assert
from ufl.algorithms.transformations import ReuseTransformer

class RestrictionPropagator(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)
        self.current_restriction = None
    
    def restricted(self, o):
        ufl_assert(self.current_restriction is None,
            "Not expecting twice restricted expression.")
        self.current_restriction = o._side
        e, = o.operands()
        r = self.visit(e)
        self.current_restriction = None
        return r

    def facet_normal(self, o):
        ufl_assert(self.current_restriction is not None, "Facet normal must be restricted.")
        #if self.current_restriction is None:
        #    return o
        return o(self.current_restriction)

    def form_argument(self, o):
        ufl_assert(self.current_restriction is not None, "Form argument must be restricted.")
        #if self.current_restriction is None:
        #    return o
        return o(self.current_restriction)

def propagate_restrictions(expression):
    ufl_assert(isinstance(expression, Expr), "Expecting Expr instance.")
    return RestrictionPropagator().visit(expression)

