"""Functions to check properties of forms and integrals."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2009-01-05"

from ufl.log import warning, debug
from ufl.assertions import ufl_assert
from ufl.common import lstr
from ufl.form import Form
from ufl.algebra import Sum, Product
from ufl.tensoralgebra import Dot
from ufl.basisfunction import BasisFunction
from ufl.algorithms.traversal import iter_expressions, pre_traversal
from ufl.algorithms.transformations import extract_basisfunction_dependencies, NotMultiLinearException

#--- Utilities for checking properties of forms ---

def is_multilinear(form):
    "Check if form is multilinear in basis function arguments."
    # An attempt at implementing is_multilinear using extract_basisfunction_dependencies.
    # TODO: This has some false negatives for "multiple configurations". (Does it still? Needs testing!)
    # TODO: FFC probably needs a variant of this which checks for some sorts of linearity
    #       in Functions as well, this should be a fairly simple extension of the current algorithm.
    try:
        for e in iter_expressions(form):
            deps = extract_basisfunction_dependencies(e)
            nargs = [len(d) for d in deps]
            if len(nargs) == 0:
                debug("This form is a functional.")
            if len(nargs) == 1:
                debug("This form is linear in %d arguments." % nargs[0])
            if len(nargs) > 1:
                warning("This form has more than one basis function "\
                    "'configuration', it has terms that are linear in %s "\
                    "arguments respectively." % str(nargs))
    
    except NotMultiLinearException, msg:
        warning("Form is not multilinear, the offending term is: %s" % msg)
        return False
    
    return True


# TODO: Remove this code if nobody needs it for anything:
#===============================================================================
# def is_multilinear(form):
#    "Check if form is multilinear."
# 
#    # Check that we get a form
#    ufl_assert(isinstance(form, Form), "Not a form: %s" % str(form))
# 
#    # Check that all operators applied to basis functions are linear
#    for e in iter_expressions(form):   
#        stack = []     
#        for o in pre_traversal(e, stack):
#            if isinstance(o, BasisFunction):
#                for operator in stack:
#                    if not operator.is_linear():
#                        warning("Nonlinear operator applied to basis function:" + str(operator))
#                        return False
# 
#    # Extract monomials
#    monomials = []
#    for e in iter_expressions(form):
#        monomials += _extract_monomials(e)
# 
#    # Extract basis functions
#    basisfunctions = set()
#    for monomial in monomials:
#        for v in monomial:
#            basisfunctions.add(v)
# 
#    # Check that each basis function appears exactly once in each monomial term
#    for monomial in monomials:
#        for v in basisfunctions:
#            if not len([w for w in monomial if w == v]) == 1:
#                warning("Basis function %s does not appear exactly once in each term." % str(v))
#                return False
# 
#    return True
#===============================================================================

