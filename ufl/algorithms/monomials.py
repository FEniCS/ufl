"""Utility algorithms for monomial representation of expressions."""

__authors__ = "Anders Logg"
__date__ = "2008-08-01 -- 2008-11-24"

# Modified by Martin Alnes, 2008

from ufl.output import ufl_assert, ufl_error
from ufl.base import Expr
from ufl.algebra import Sum, Product
from ufl.basisfunction import BasisFunction
from ufl.function import Function
from ufl.form import Form
from ufl.algorithms.traversal import iter_expressions


#--- Utilities to extract information from an expression ---

def extract_monomials(expression, indent=""):
    "Extract monomial representation of expression (if possible)."

    # FIXME: Not yet working, need to include derivatives, integrals etc

    ufl_assert(isinstance(expression, Form) or isinstance(expression, Expr), "Expecting UFL form or expression.")

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
        ufl_error("is_linear has been removed, only had a partial and flawed implementation.")

        if not e.is_linear():
            ufl_error("Operator is nonlinear, unable to extract monomials: " + str(e))
            
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
            ufl_error("Don't know how to handle expression: %s", str(e))

    return m

