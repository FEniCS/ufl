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
from ufl.algorithms.transformations import ReuseTransformer, apply_transformer
from ufl.algorithms.ad import expand_derivatives

# Martin to Marie and Anders:
#    I wouldn't trust these functions,
#    things have changed since Anders first wrote them
#    and they only seem like sketches anyway.
# Note from Anders to Martin: Work in progress

# Exception raised when monomial extraction fails
class MonomialException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class MonomialTransformer(ReuseTransformer):

    def __init__(self):
        print "Creating TestTransformer"
        ReuseTransformer.__init__(self)
        
    def expr(self, o, *ops):
        raise MonomialException, ("No handler defined for %s." % o._uflclass.__name__)

    def sum(self, o, monomials0, monomials1):
        monomials = monomials0 + monomials1
        return monomials

    def product(self, o, monomials0, monomials1):
        monomials = []
        for monomial0 in monomials0:
            for monomial1 in monomials1:
                monomials.append(monomial0 + monomial1)
        return monomials
        
    def index_sum(self, o, monomials, index):
        return monomials

    def indexed(self, o, monomials, indices):
        #operators["c"].append(indices)
        return monomials
    
    def component_tensor(self, o, monomials, indices):
        return monomials
        
    def spatial_derivative(self, o, monomials, index):
        for monomial in monomials:
            if len(monomial) > 1:
                error("Expecting a single basis function.")
            for v in monomial:
                v[1]["d"] = index
        return monomials

    def basis_function(self, o):
        v = [o, {"d": []}]
        monomials = [[v]]
        return monomials

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
        integrand = expand_derivatives(integrand)

        print ""
        print "Original integrand: " + str(integrand)

        monomials = apply_transformer(integrand, MonomialTransformer())

        #print "m =", measure
        #print "I1 =", integral.integrand
        #print "I2 =", integrand

        #try:
        #    monomials = _extract_monomials(integrand)
        #except MonomialException:
        #    warning("Unable to extract monomial")
        #    return None

    # Print monomial
    print ""
    print "Number of terms:", len(monomials)
    for (i, m) in enumerate(monomials):
        print "term", i
        for (v, ops) in m:
            print "  ", v, ops

    return
