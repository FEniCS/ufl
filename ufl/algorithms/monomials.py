"""Utility algorithms for monomial representation of expressions."""

__authors__ = "Anders Logg"
__date__ = "2008-08-01 -- 2009-01-09"

# Modified by Martin Alnes, 2008

from copy import deepcopy

from ufl.log import info, error, warning, begin, end, set_level, INFO
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.algebra import Sum, Product
from ufl.indexsum import IndexSum
from ufl.indexed import Indexed
from ufl.tensors import ComponentTensor
from ufl.tensoralgebra import Dot
from ufl.basisfunction import BasisFunction
from ufl.function import Function
from ufl.differentiation import SpatialDerivative
from ufl.form import Form
from ufl.algorithms.traversal import iter_expressions
from ufl.algorithms.transformations import expand_compounds

# Martin to Marie and Anders:
#    I wouldn't trust these functions,
#    things have changed since Anders first wrote them
#    and they only seem like sketches anyway.
# Note from Anders to Martin: Work in progress

# Exception raised when monomial extraction fails
class MonomialException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

#--- Utilities to extract information from an expression ---

def extract_monomials(form, indent=""):
    """Extract monomial representation of form (if possible). When
    successful, the form is represented as a sum of products of scalar
    components of basis functions of derivatives of basis functions.
    The sum of products is represented as a tuple of tuples of basis
    functions. If unsuccessful, None is returned, indicating that it
    is not possible to extract a monomial representation of the form."""

    # FIXME: In progress

    # Check that we get a Form
    ufl_assert(isinstance(form, Form), "Expecting a UFL form.")

    set_level(INFO)

    print ""
    print "Extracting monomials"
    print "--------------------"
    print "a = " + str(form)

    monomials = []    
    for integral in form.cell_integrals():

        # Get measure and integrand
        measure = integral.measure()
        integrand = integral.integrand()

        # Expand compounds
        integrand = expand_compounds(integrand)

        print "m =", measure
        print "I1 =", integral.integrand
        print "I2 =", integrand

        try:
            monomials = _extract_monomials(integrand)
        except MonomialException:
            warning("Unable to extract monomial")
            return None

    # Print monomial
    print ""
    print "Number of terms:", len(monomials)
    for (i, m) in enumerate(monomials):
        print "term", i
        for (v, ops) in m:
            print "  ", v, ops

    return

def _extract_monomials(expr, operators={"i": [], "c": [], "d": []}):
    "Recursively extract monomials from expression."

    # Check that we get an Expression
    ufl_assert(isinstance(expr, Expr), "Expecting a UFL expression.")

    # Make copy to avoid recursively accumulating operators
    operators = deepcopy(operators)

    # Handle expression
    if isinstance(expr, Sum):
        begin("Sum")
        v, w = expr.operands()
        monomials = _extract_monomials(v, operators) + _extract_monomials(w, operators)
    elif isinstance(expr, Product):
        begin("Product")
        v, w = expr.operands()
        monomials = []
        for m0 in _extract_monomials(v, operators):
            for m1 in _extract_monomials(w, operators):
                monomials.append(m0 + m1)
        monomials = tuple(monomials)
    elif isinstance(expr, IndexSum):
        begin("IndexSum, ignoring for now")
        (summand, index) = expr.operands()
        monomials = _extract_monomials(summand, operators)
    elif isinstance(expr, Indexed):
        begin("Indexed")
        (expression, indices) = expr.operands()
        operators["c"].append(indices)
        monomials = _extract_monomials(expression, operators)
    elif isinstance(expr, ComponentTensor):
        begin("ComponentTensor, ignoring for now")
        (expression, indices) = expr.operands()
        monomials = _extract_monomials(expression, operators)
    elif isinstance(expr, SpatialDerivative):
        begin("SpatialDerivative")
        (expression, index) = expr.operands()
        operators["d"].append(index)
        monomials = _extract_monomials(expression, operators)
    elif isinstance(expr, BasisFunction):
        begin("BasisFunction")
        operators["i"] = expr.count()
        v = (expr, operators)
        info("v = " + str(expr))
        info("ops = " + str(operators))
        monomials = ((v,),)
    else:
        raise MonomialException, "Unhandled expression: " + str(expr)

    end()

    return monomials
