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
from ufl.algorithms.printing import tree_format
from ufl.algorithms.renumbering import renumber_indices

# Martin to Marie and Anders:
#    I wouldn't trust these functions,
#    things have changed since Anders first wrote them
#    and they only seem like sketches anyway.
# Note from Anders to Martin: Work in progress

# Exception raised when monomial extraction fails
class MonomialException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class MonomialFactor:

    def __init__(self, arg=None):
        if isinstance(arg, MonomialFactor):
            self.basis_function = arg.basis_function
            self.component = [c for c in arg.component]
            self.derivative = [d for d in arg.derivative]
        elif isinstance(arg, BasisFunction):
            self.basis_function = arg
            self.component = []
            self.derivative = []
        elif arg is None:
            self.basis_function = None
            self.component = []
            self.derivative = []
        else:
            raise MonomialException, ("Unable to create monomial from expression: " + str(arg))

    def dx(self, index):
        v = MonomialFactor(self)
        v.derivative.append(index)
        return v

    def __getitem__(self, index):
        if len(self.component) > 0:
            raise MonomialException, "Basis function already indexed."
        v = MonomialFactor(self)
        v.component.append(index)
        return v

    def __str__(self):
        c = ""
        if len(self.component) > 0:
            c = "[%s]" % ", ".join(str(c) for c in self.component)
        d0 = ""
        if len(self.derivative) > 0:
            d0 = "(" + " ".join("d/dx_%s" % str(d) for d in self.derivative) + " "
            d1 = ")"
        return d0 + str(self.basis_function) + d1 + c

class Monomial:
    
    def __init__(self, arg=None):
        if isinstance(arg, Monomial):
            self.factors = [MonomialFactor(v) for v in arg.factors]
        elif isinstance(arg, MonomialFactor):
            self.factors = [MonomialFactor(arg)]
        elif arg is None:
            self.factors = []
        else:
            raise MonomialException, ("Unable to create monomial from expression: " + str(arg))

    def dx(self, index):
        if len(self.factors) > 1:
            raise MonomialException, "Expecting a single basis function."
        m = Monomial(self)
        m.factors[0] = m.factors[0].dx(index)
        return m

    def __getitem__(self, index):
        if len(self.factors) > 1:
            raise MonomialException, "Expecting a single basis function."
        m = Monomial(self)
        m.factors[0] = m.factors[0][index]
        return m

    def __mul__(self, other):
        m = Monomial()
        m.factors = self.factors + other.factors
        return m

    def __str__(self):
        return "*".join(str(v) for v in self.factors)

class MonomialTransformer(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)
    
    def expr(self, o, *ops):
        raise MonomialException, ("No handler defined for expression %s." % o._uflclass.__name__)

    def terminal(self, o):
        raise MonomialException, ("No handler defined for terminal %s." % o._uflclass.__name__)

    def variable(self, o):
        raise MonomialException, ("No handler defined for variable %s." % o._uflclass.__name__)

    def sum(self, o, monomials0, monomials1):
        monomials = monomials0 + monomials1
        return monomials

    def product(self, o, monomials0, monomials1):
        m = []
        for monomial0 in monomials0:
            for monomial1 in monomials1:
                m.append(monomial0*monomial1)
        return m
        
    def index_sum(self, o, monomials, index):
        return monomials

    def indexed(self, o, monomials, index):
        m = []
        for monomial in monomials:
            m.append(monomial[index])
        return m
    
    def component_tensor(self, o, monomials, indices):
        return monomials
        
    def spatial_derivative(self, o, monomials, index):
        m = []
        for monomial in monomials:
            m.append(monomial.dx(index))
        return m

    def multi_index(self, o):
        print "Ignoring MultiIndex terminal for now"
        return o

    def basis_function(self, o):
        return [Monomial(MonomialFactor(o))]

#    def function(self, o):
#        v = [BasisFunction(o.element()), self.empty_operators()]
#        monomials = [[v]]
#        return monomials

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

        print ""
        print "Original integrand:"
        print integrand
        print ""

        # Expand compounds
        integrand = expand_derivatives(integrand)

        # Renumber indices
        integrand = renumber_indices(integrand)

        print ""
        print "Transformed integrand:"
        print integrand
        print ""

        #print tree_format(integrand)

        # Extract monomial representation
        monomials = apply_transformer(integrand, MonomialTransformer())

        #print "m =", measure
        #print "I1 =", integral.integrand
        #print "I2 =", integrand

        #try:
        #    monomials = _extract_monomials(integrand)
        #except MonomialException:
        #    warning("Unable to extract monomial")
        #    return None

    # Print monomial representation
    print ""
    print "Number of terms:", len(monomials)
    for monomial in monomials:
        print "  ", monomial

    return
