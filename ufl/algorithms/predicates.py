"""Functions to check properties of forms and integrals."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-09-28"

from ..output import ufl_error, ufl_warning, ufl_assert, ufl_info
from ..common import lstr
from ..form import Form
from ..base import Terminal
from ..basisfunction import BasisFunction
from ..algebra import Sum, Product
from .traversal import iter_expressions, traversal

#--- Utilities for checking properties of forms ---

def is_multilinear(form):
    "Check if form is multilinear"

    ufl_assert(isinstance(form, Form), "Not a form: %s" % str(form))
    ufl_warning("is_multilinear does not work yet (in preparation)")

    print ""
    print form
    print ""

    for e in iter_expressions(form):
        for (o, stack) in traversal(e):
            print o
    print ""

    # Iterate over integrands
    #for e in iter_expressions(form):

        # Iterate over operands
    #    for o in post_traversal(e):
            #print o, ":", stack
    #        print o
            #if isinstance(o, BasisFunction):
            #    print o
            #elif isinstance(o, Sum):
            #    print "+"
            #elif isinstance(o, Product):
            #    print "*"
            #if not o.is_linear():
            #    return False

    print ""

    #def foo(o):
    #    print "hej:", o
    #
    #pre_walk(form, foo)
    #
    #print ""
    

    # FIXME: Check arity of each term, must be equal

    return True
