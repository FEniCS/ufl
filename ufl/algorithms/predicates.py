"""Functions to check properties of forms and integrals."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2008-09-30"

from ..output import ufl_assert
from ..common import lstr
from ..form import Form
from ..algebra import Sum, Product
from ..basisfunction import BasisFunction
from .traversal import iter_expressions, traversal

#--- Utilities for checking properties of forms ---

def is_multilinear(form):
    "Check if form is multilinear."

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Not a form: %s" % str(form))

    # Check that all operators applied to basis functions are linear
    for e in iter_expressions(form):        

        # FIXME: This works
        for (o, stack) in traversal(e, []):

        # FIXME: This does not work
        #for (o, stack) in traversal(e,):

            
            if isinstance(o, BasisFunction):
                for operator in stack:
                    if not operator.is_linear():
                        # FIXME: Use ufl_info or ufl_debug here
                        print "Nonlinear operator applied to basis function:", str(operator)
                        return False

    # Extract monomials
    monomials = []
    for e in iter_expressions(form):
        monomials += _extract_monomials(e)

    # Extract basis functions
    basisfunctions = set()
    for monomial in monomials:
        for v in monomial:
            basisfunctions.add(v)

    # Check that each basis function appears exactly once in each monomial term
    for monomial in monomials:
        for v in basisfunctions:
            if not len([w for w in monomial if w == v]) == 1:
                # FIXME: Use ufl_info or ufl_debug here
                print "Basis function %s does not appear exactly once in each term." % str(v)
                return False

    return True

def _extract_monomials(e):
    "Extract monomial terms (ignoring all operators except + and -)"
    
    operands = e.operands()
    monomials = []
    if isinstance(e, Sum):
        ufl_assert(len(operands) == 2, "Strange, expecting two terms.")
        monomials += _extract_monomials(operands[0])
        monomials += _extract_monomials(operands[1])
    elif isinstance(e, Product):
        ufl_assert(len(operands) == 2, "Strange, expecting two factors.")
        for m0 in _extract_monomials(operands[0]):
            for m1 in _extract_monomials(operands[1]):
                monomials.append(m0 + m1)
    elif isinstance(e, BasisFunction):
        monomials.append((e,))

    return monomials
