"""Utility algorithms for monomial representation of expressions."""

__authors__ = "Anders Logg"
__date__ = "2008-08-01 -- 2009-03-04"

# Modified by Martin Alnes, 2008

from copy import deepcopy

from ufl.log import info, error, warning, begin, end, set_level, INFO
from ufl.assertions import ufl_assert
from ufl.expr import Expr
from ufl.algebra import Sum, Product
from ufl.indexsum import IndexSum
from ufl.indexed import Indexed
from ufl.indexing import Index, MultiIndex
from ufl.tensors import ComponentTensor
from ufl.tensoralgebra import Dot
from ufl.basisfunction import BasisFunction
from ufl.function import Function
from ufl.constantvalue import ScalarValue
from ufl.differentiation import SpatialDerivative
from ufl.form import Form
from ufl.algorithms.traversal import iter_expressions
from ufl.algorithms.transformations import purge_list_tensors
from ufl.algorithms.transformations import ReuseTransformer, apply_transformer
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.printing import tree_format
from ufl.algorithms.renumbering import renumber_indices

# Exception raised when monomial extraction fails
class MonomialException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class MonomialFactor:

    def __init__(self, arg=None):
        if isinstance(arg, MonomialFactor):
            self.function = arg.function
            self.component = arg.component
            self.derivative = arg.derivative
        elif isinstance(arg, BasisFunction) or isinstance(arg, Function):
            self.function = arg
            self.component = []
            self.derivative = []
        elif arg is None:
            self.function = None
            self.component = []
            self.derivative = []
        else:
            raise MonomialException, ("Unable to create monomial from expression: " + str(arg))

    def apply_derivative(self, indices):
        self.derivative += indices

    def replace_indices(self, old_indices, new_indices):
        if old_indices is None:
            self.component = new_indices
        else:
            _replace_indices(self.component, old_indices, new_indices)
            _replace_indices(self.derivative, old_indices, new_indices)

    def __str__(self):
        c = ""
        if len(self.component) == 0:
            c = ""
        else:
            c = "[%s]" % ", ".join(str(c) for c in self.component)
        if len(self.derivative) == 0:
            d0 = ""
            d1 = ""
        else:
            d0 = "(" + " ".join("d/dx_%s" % str(d) for d in self.derivative) + " "
            d1 = ")"
        return d0 + str(self.function) + c + d1

class Monomial:
    
    def __init__(self, arg=None):
        if isinstance(arg, Monomial):
            self.float_value = arg.float_value
            self.factors = [MonomialFactor(v) for v in arg.factors]
            self.index_slots = arg.index_slots
        elif isinstance(arg, MonomialFactor) or isinstance(arg, BasisFunction) or isinstance(arg, Function):
            self.float_value = 1.0
            self.factors = [MonomialFactor(arg)]
            self.index_slots = None
        elif isinstance(arg, ScalarValue):
            self.float_value = float(arg)
            self.factors = []
            self.index_slots = None
        elif arg is None:
            self.float_value = 1.0
            self.factors = []
            self.index_slots = None
        else:
            raise MonomialException, ("Unable to create monomial from expression: " + str(arg))

    def apply_derivative(self, indices):
        if not len(self.factors) == 1:
            raise MonomialException, "Expecting a single factor."
        self.factors[0].apply_derivative(indices)

    def apply_tensor(self, indices):
        if not self.index_slots is None:
            raise MonomialException, "Expecting scalar-valued expression."
        self.index_slots = indices
    
    def apply_indices(self, indices):
        print "Applying indices:", self.index_slots, "-->", indices
        for v in self.factors:
            v.replace_indices(self.index_slots, indices)
        self.index_slots = None

    def __mul__(self, other):
        m = Monomial()
        m.float_value = self.float_value * other.float_value
        m.factors = self.factors + other.factors
        return m

    def __str__(self):
        if self.float_value == 1.0:
            float_value = ""
        else:
            float_value = "%g * " % self.float_value
        return float_value + " * ".join(str(v) for v in self.factors)

class MonomialForm:

    def __init__(self, arg=None):
        if isinstance(arg, MonomialForm):
            self.monomials = [Monomial(m) for m in arg.monomials]
        elif arg is None:
            self.monomials = []
        else:
            self.monomials = [Monomial(arg)]

    def apply_derivative(self, indices):
        for m in self.monomials:
            m.apply_derivative(indices)

    def apply_tensor(self, indices):
        for m in self.monomials:
            m.apply_tensor(indices)

    def apply_indices(self, indices):
        for m in self.monomials:
            m.apply_indices(indices)

    def __add__(self, other):
        form = MonomialForm()
        form.monomials = [Monomial(m) for m in self.monomials] + [Monomial(m) for m in other.monomials]
        return form

    def __mul__(self, other):
        form = MonomialForm()
        for m0 in self.monomials:
            for m1 in other.monomials:
                form.monomials.append(m0 * m1)
        return form

    def __str__(self):
        return " + ".join(str(m) for m in self.monomials)

class MonomialTransformer(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)
    
    def expr(self, o, *ops):
        raise MonomialException, ("No handler defined for expression %s." % o._uflclass.__name__)

    def terminal(self, o):
        raise MonomialException, ("No handler defined for terminal %s." % o._uflclass.__name__)

    def variable(self, o):
        return self.visit(o.expression())

    def sum(self, o, form0, form1):
        print "\nSum"
        form = form0 + form1
        print "Result:", form
        return form

    def product(self, o, form0, form1):
        print "\nProduct: [%s] * [%s]" % (str(form0), str(form1))
        form = form0 * form1
        print "Result:", form
        return form

    def index_sum(self, o, form, index):
        print "\nIgnoring IndexSum expression for now"
        print "Result:", form
        return form

    def indexed(self, o, form, indices): 
        print "\nIndexed", form, indices
        form = MonomialForm(form)
        form.apply_indices(indices)
        print "Result:", form
        return form

    def component_tensor(self, o, form, indices):
        print "\nComponentTensor", form, indices
        form = MonomialForm(form)
        form.apply_tensor(indices)
        print "Result:", form
        return form

    def spatial_derivative(self, o, form, indices):
        print "\nSpatialDerivative", form, indices
        form = MonomialForm(form)
        form.apply_derivative(indices)
        print "Result:", form
        return form

    def multi_index(self, multi_index):
        print "\nMultiIndex"
        indices = [index for index in multi_index]
        print indices
        return indices

    def index(self, o):
        raise MonomialException, "Not expecting to see an Index terminal."

    def basis_function(self, v):
        print "\nBasisFunction", v
        form = MonomialForm(v)
        print "Result:", form
        return form

    def function(self, v):
        print "\nFunction", v
        form = MonomialForm(v)
        print "Result:", form
        return form

    def scalar_value(self, x):
        print "\nScalarValue", x
        form = MonomialForm(x)
        print "Result:", form
        return form

def extract_monomials(form, indent=""):
    """Extract monomial representation of form (if possible). When
    successful, the form is represented as a sum of products of scalar
    components of basis functions of derivatives of basis functions.
    The sum of products is represented as a tuple of tuples of basis
    functions. If unsuccessful, MonomialException is raised."""

    # FIXME: In progress

    # Check that we get a Form
    ufl_assert(isinstance(form, Form), "Expecting a UFL form.")

    set_level(INFO)

    print "Extracting monomials"
    print "--------------------"
    print "a = " + str(form)
    
    # Extract processed form
    form_data = form.form_data()
    form = form_data.form

    monomials = []    
    for integral in form.cell_integrals():

        # Get measure and integrand
        measure = integral.measure()
        integrand = integral.integrand()

        print ""
        print "Original integrand:"
        print integrand
        print ""

        # Purge list tensors from expression tree
        integrand = purge_list_tensors(integrand)

        print ""
        print "Transformed integrand:"
        print integrand
        print ""

        print tree_format(integrand)

        # Extract monomial representation if possible
        monomials = apply_transformer(integrand, MonomialTransformer())

        #print "m =", measure
        #print "I1 =", integral.integrand
        #print "I2 =", integrand

        return monomials

    # Print monomial representation
    print ""
    print "Number of terms:", len(monomials)
    for monomial in monomials:
        print "  ", monomial

    return monomials

def _replace_indices(indices, old_indices, new_indices):
    "Handle replacement of subsets of multi indices."

    # Old and new indices must match
    if not len(old_indices) == len(new_indices):
        raise MonomialException, "Unable to replace indices, mismatching index dimensions."

    # Build index map
    index_map = {}
    for (i, index) in enumerate(old_indices):
        index_map[index] = new_indices[i]

    # Check all indices and replace
    for (i, index) in enumerate(indices):
        if index in old_indices:
            indices[i] = index_map[index]
