"""Functions to check properties of forms and integrals."""


__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-10-24"

from ufl.output import ufl_assert, ufl_warning, ufl_debug
from ufl.common import lstr
from ufl.form import Form
from ufl.algebra import Sum, Product
from ufl.tensoralgebra import Dot
from ufl.basisfunction import BasisFunction
from ufl.algorithms.traversal import iter_expressions, traversal
from ufl.algorithms.analysis import extract_basisfunction_dependencies, NotMultiLinearException

#--- Utilities for checking properties of forms ---

#===============================================================================
# def is_multilinear(form):
#    "Check if form is multilinear."
# 
#    # Check that we get a form
#    ufl_assert(isinstance(form, Form), "Not a form: %s" % str(form))
# 
#    # Check that all operators applied to basis functions are linear
#    for e in iter_expressions(form):        
#        for (o, stack) in traversal(e):
#            if isinstance(o, BasisFunction):
#                for operator in stack:
#                    if not operator.is_linear():
#                        ufl_warning("Nonlinear operator applied to basis function:" + str(operator))
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
#                ufl_warning("Basis function %s does not appear exactly once in each term." % str(v))
#                return False
# 
#    return True
#===============================================================================

def is_multilinear(form):
    "Check if form is multilinear in basis function arguments."
    # An attempt at implementing is_multilinear using extract_basisfunction_dependencies.
    # TODO: FFC probably needs a variant of this which checks for linearity in Functions as well.
    try:
        for e in iter_expressions(form):
            deps = extract_basisfunction_dependencies(e)
            if len(deps) != 1:
                ufl_warning("This form has more than one basis function 'configuration', i.e. it could have both linear and bilinear terms.")
    except NotMultiLinearException, msg:
        ufl_warning("Form is not multilinear, the offending term is: %s" % msg)
        return False
    return True

def _extract_monomials(e):
    "Extract monomial terms (ignoring all operators except + and -)"
    
    operands = e.operands()
    monomials = []
    if isinstance(e, Sum):
        for o in operands:
            monomials += _extract_monomials(o)
    # FIXME: Does this make sense (treating Dot like Product)? No.
    elif isinstance(e, Product) or isinstance(e, Dot):
        ufl_assert(len(operands) == 2, "Strange, expecting two factors.")
        for m0 in _extract_monomials(operands[0]):
            for m1 in _extract_monomials(operands[1]):
                monomials.append(m0 + m1)
    elif len(operands) == 2:
        ufl_warning("Unknown binary operator, don't know how to handle.")
    elif len(operands) == 1: # FIXME: This won't be right for lots of operators... Should at least throw errors for unsupported types.
        monomials += _extract_monomials(operands[0])
    elif isinstance(e, BasisFunction):
        monomials.append((e,))

    return monomials
