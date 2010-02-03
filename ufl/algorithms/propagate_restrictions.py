"Algorithms related to restrictions."

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-05-14 -- 2009-06-08"

from ufl.expr import Expr
from ufl.assertions import ufl_assert
from ufl.algorithms.transformations import Transformer, ReuseTransformer

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

    def variable(self, o):
        ufl_assert(self.current_restriction is not None, "Form argument must be restricted.")
        #if self.current_restriction is None:
        #    return o
        return o(self.current_restriction)

class RestrictionChecker(Transformer):
    def __init__(self, require_restriction):
        Transformer.__init__(self)
        self.current_restriction = None
        self.require_restriction = require_restriction

    def expr(self, o):
        pass

    def restricted(self, o):
        ufl_assert(self.current_restriction is None,
            "Not expecting twice restricted expression.")
        self.current_restriction = o._side
        e, = o.operands()
        self.visit(e)
        self.current_restriction = None

    def facet_normal(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None, "Facet normal must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None, "Restrictions are only allowed for interior facet integrals.")

    def form_argument(self, o):
        if self.require_restriction:
            ufl_assert(self.current_restriction is not None, "Form argument must be restricted in interior facet integrals.")
        else:
            ufl_assert(self.current_restriction is None, "Restrictions are only allowed for interior facet integrals.")

def propagate_restrictions(expression):
    ufl_assert(isinstance(expression, Expr), "Expecting Expr instance.")
    return RestrictionPropagator().visit(expression)

def check_restrictions(expression, require_restriction):
    ufl_assert(isinstance(expression, Expr), "Expecting Expr instance.")
    return RestrictionChecker(require_restriction).visit(expression)

