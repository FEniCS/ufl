"""Utility algorithms for monomial representation of expressions."""

__authors__ = "Anders Logg"
__date__ = "2008-08-01 -- 2009-01-09"

# Modified by Martin Alnes, 2008

from ufl.log import ufl_assert, error, warning
from ufl.expr import Expr
from ufl.algebra import Sum, Product
from ufl.tensoralgebra import Dot
from ufl.basisfunction import BasisFunction
from ufl.function import Function
from ufl.form import Form
from ufl.algorithms.traversal import iter_expressions


# Martin to Marie and Anders:
#    I wouldn't trust these functions,
#    things have changed since Anders first wrote them
#    and they only seem like sketches anyway.


#--- Utilities to extract information from an expression ---

def extract_monomials(expression, indent=""):
    "Extract monomial representation of expression (if possible)."

    # TODO: Not yet working, need to include derivatives, integrals etc

    ufl_assert(isinstance(expression, Form) or isinstance(expression, Expr),
        "Expecting UFL form or expression.")

    # Iterate over expressions
    m = []

    print ""
    print "Extracting monomials"

    #cell_integrals = expression.cell_integrals()
    #print cell_integrals
    #print dir(cell_integrals[0].)
    #integrals

    for e in iter_expressions(expression):

        # Check for linearity
        error("is_linear has been removed, only had a partial and flawed implementation.")

        if not e.is_linear():
            error("Operator is nonlinear, unable to extract monomials: " + str(e))
            
        print indent + "e =", e, str(type(e))
        operands = e.operands()
        if isinstance(e, Sum):
            ufl_assert(len(operands) == 2, "Strange, expecting two terms.")
            m += extract_monomials(operands[0], indent + "  ")
            m += extract_monomials(operands[1], indent + "  ")
        elif isinstance(e, Product):
            ufl_assert(len(operands) == 2, "Strange, expecting two factors.")
            for m0 in extract_monomials(operands[0], indent + "  "):
                for m1 in extract_monomials(operands[1], indent + "  "):
                    m.append(m0 + m1)
        elif isinstance(e, BasisFunction):
            m.append((e,))
        elif isinstance(e, Function):
            m.append((e,))
        else:
            print "type =", type(e)
            print "free indices =", e.free_indices()
            error("Don't know how to handle expression: %s", str(e))

    return m

def _extract_monomials(e): 
    "Extract monomial terms (ignoring all operators except + and -)"
    
    operands = e.operands()
    monomials = []
    if isinstance(e, Sum):
        for o in operands:
            monomials += _extract_monomials(o)
    # TODO: Does this make sense (treating Dot like Product)? No. Use expand_compounds first, then there are no Dot, Inner, Outer, Cross, etc.
    elif isinstance(e, Product) or isinstance(e, Dot):
        ufl_assert(len(operands) == 2, "Strange, expecting two factors.")
        for m0 in _extract_monomials(operands[0]):
            for m1 in _extract_monomials(operands[1]):
                monomials.append(m0 + m1)
    elif len(operands) == 2:
        warning("Unknown binary operator, don't know how to handle.")
    elif len(operands) == 1: # TODO: This won't be right for lots of operators... Should at least throw errors for unsupported types.
        monomials += _extract_monomials(operands[0])
    elif isinstance(e, BasisFunction):
        monomials.append((e,))

    return monomials

