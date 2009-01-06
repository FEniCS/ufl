"""This module defines automatic differentiation utilities."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-19-- 2008-12-22"

from ufl.output import ufl_assert, ufl_error, ufl_warning
from ufl.common import product, unzip, UFLTypeDefaultDict, subdict, mergedicts

# All classes:
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.zero import Zero
from ufl.form import Form
from ufl.integral import Integral
from ufl.scalar import FloatValue, IntValue
from ufl.variable import Variable
from ufl.finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ufl.basisfunction import BasisFunction, BasisFunctions
from ufl.function import Function, Constant
from ufl.indexing import MultiIndex, Indexed, Index
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional

from ufl.classes import ScalarValue, Zero, Identity, Constant, VectorConstant, TensorConstant

# Lists of all Expr classes
#from ufl.classes import ufl_classes, terminal_classes, nonterminal_classes
from ufl.classes import terminal_classes
from ufl.operators import dot, inner, outer, lt, eq, conditional
from ufl.operators import sqrt, exp, ln, cos, sin
from ufl.algorithms.traversal import iter_expressions
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.transformations import expand_compounds, Transformer, transform, transform_integrands

# FIXME: Need algorithm to apply AD to all kinds of derivatives!
#        In particular, SpatialDerivative, VariableDerivative and functional derivative.

# FIXME: Need some cache structures and callback to custum diff routine to implement diff with variable
# - Check for diff of variable in some kind of cache
# - If not found, apply diff to variable expression 
# - Add variable for differentated expression to cache

# FIXME: Missing rules for:
# Cross, Determinant, Cofactor, f(x)**g(x)
# FIXME: Could apply as_basic to Compound objects with no rule before differentiating

def is_spatially_constant(o):
    return (isinstance(o, Terminal) and o.cell() is None) or isinstance(o, Constant)

_0 = Zero()
_1 = IntValue(1)

class AD(Transformer):
    def __init__(self, spatial_dim):
        Transformer.__init__(self)
        self._spatial_dim = spatial_dim
    
    def expr(self, o, *ops):
        ufl_error("Missing AD handler for type %s" % str(type(o)))
    
    def variable(self, o):
        ufl_error("How to handle derivative of variable depends on context. You must supply a customized rule!")
    
    def terminal(self, o):
        """Terminal objects are assumed independent of the differentiation
        variable by default, and simply 'lifted' to the pair (o, 0).
        Depending on the context, override this with custom rules for
        non-zero derivatives."""
        return (o, Zero(o.shape()))
    
    def commute(self, o, a):
        "This should work for all single argument operators that commute with d/dw."
        aprime = a[1]
        return (o, type(o)(aprime))
    
    transposed = commute
    trace = commute
    deviatoric = commute
    div  = commute
    curl = commute
    rot  = commute
    
    def multi_index(self, o):
        return (o, None) # oprime here should never be used
    
    def indexed(self, o, A, ii):
        return (o, A[1][ii[0]])
    
    def list_tensor(self, o, *ops):
        opprimes = [op[1] for op in ops]
        return (o, ListTensor(*opprimes))
    
    def component_tensor(self, o, A, ii):
        if isinstance(A[1], Zero):
            fi = o.free_indices()
            fid = subdict(o.index_dimensions(), fi)
            return (o, Zero(o.shape(), fi, fid))
        return (o, ComponentTensor(A[1], ii[0]) )
    
    def sum(self, o, *ops):
        return (sum((op[0] for op in ops[1:]), ops[0][0]),
                sum((op[1] for op in ops[1:]), ops[0][1]))
    
    def product(self, o, *ops):
        fi = o.free_indices()
        fid = subdict(o.index_dimensions(), fi)
        fp = Zero(o.shape(), fi, fid)
        ops0, ops1 = unzip(ops)
        for (i,op) in enumerate(ops):
            # replace operand i with its differentiated value 
            fpoperands = ops0[:i] + [ops1[i]] + ops0[i+1:]
            # simplify by ignoring ones
            fpoperands = [fpop for fpop in fpoperands if not fpop == 1]
            # simplify if there are zeros in the product
            if not any(isinstance(fpop, Zero) for fpop in fpoperands):
                fp += product(fpoperands)
        return (o, fp)
    
    def division(self, o, a, b):
        f, fp = a
        g, gp = b
        return (o, (fp-f*gp/g)/g)
        #return (o, (fp*g-f*gp)/g**2)
    
    def power(self, o, a, b):
        f, fp = a
        g, gp = b
        ufl_assert(not (f.shape() or g.shape()), "Expecting scalar expressions f,g in f**g.")
        # o = f**g
        f_const = isinstance(fp, Zero)
        g_const = isinstance(gp, Zero)
        # Case: o = const ** const = const
        if f_const and g_const:
            return (o, _0)
        # Case: o = f(x) ** const
        if g_const:
            # o' = g f'(x) f(x)**(g-1)
            if isinstance(g, Zero) or isinstance(f, Zero) or f_const:
                return (o, _0)
            return (o, g*fp*f**(g-1.0))
        # Case: o = f ** g(x)
        if isinstance(fp, Zero):
            return (o, gp*ln(f)*o)
        ufl_error("diff_power not implemented for case d/dx [ f(x)**g(x) ].")
        oprime = None # TODO
        return (o, oprime)
    
    def abs(self, o, a):
        f, fprime = a
        oprime = conditional(eq(f, 0),
                             0,
                             conditional(lt(f, 0), -fprime, fprime))
        return (o, oprime)
    
    # --- Compound operators
    
    def outer(self, o, a, b): # COMPOUND
        a, ap = a
        b, bp = b
        return (o, outer(ap, b) + outer(a, bp))
    
    def inner(self, o, a, b): # COMPOUND
        a, ap = a
        b, bp = b
        return (o, inner(ap, b) + inner(a, bp))
    
    def dot(self, o, a, b): # COMPOUND
        a, ap = a
        b, bp = b
        return (o, dot(ap, b) + dot(a, bp))
    
    def cross(self, o, a, b): # COMPOUND
        ufl_error("Derivative of cross product not implemented, apply expand_compounds before AD.")
        u, up = a
        v, vp = b
        oprime = None # TODO
        return (o, oprime)
    
    def determinant(self, o, a): # COMPOUND
        ufl_error("Derivative of determinant not implemented, apply expand_compounds before AD.")
        A, Ap = a
        oprime = None # TODO
        return (o, oprime)
    
    def cofactor(self, o, a): # COMPOUND
        ufl_error("Derivative of cofactor not implemented, apply expand_compounds before AD.")
        A, Ap = a
        #cofacA_prime = detA_prime*Ainv + detA*Ainv_prime
        oprime = None # TODO
        return (o, oprime)
    
    def inverse(self, o, a): # COMPOUND
        """Derivation:
        0 = d/dx [Ainv*A] = Ainv' * A + Ainv * A'
        Ainv' * A = - Ainv * A'
        Ainv' = - Ainv * A' * Ainv
        """
        A, Ap = a
        return (o, -o*Ap*o)
    
    # --- Mathfunctions
    
    def math_function(self, o, a):
        f, fp = a
        return (o, _0) if isinstance(fp, Zero) else (o, 0.5*fp/sqrt(f))
    
    def sqrt(self, o, a):
        f, fp = a
        return (o, _0) if isinstance(fp, Zero) else (o, 0.5*fp/sqrt(f))
    
    def exp(self, o, a):
        f, fp = a
        return (o, _0) if isinstance(fp, Zero) else (o, fp*exp(f))
    
    def ln(self, o, a):
        f, fp = a
        ufl_assert(not isinstance(f, Zero), "Division by zero.")
        return (o, _0) if isinstance(fp, Zero) else (o, fp/f)
    
    def cos(self, o, a):
        f, fp = a
        return (o, _0) if isinstance(fp, Zero) else (o, -fp*sin(f))
    
    def sin(self, o, a):
        f, fp = a
        if isinstance(fp, Zero): return (o, _0)
        return (o, fp*cos(f))
    
    # --- Restrictions
    
    def positive_restricted(self, o, a):
        f, fp = a
        return (o, fp('+')) # TODO: What is d(v+)/dw ? Assuming here that restriction and differentiation commutes.

    def negative_restricted(self, o, a):
        f, fp = a
        return (o, fp('-')) # TODO: What is d(v-)/dw ? Assuming here that restriction and differentiation commutes.
    
    # --- Conditionals
    
    def condition(self, o, l, r):
        if any(not isinstance(op[1], Zero) for op in (l, r)):
            ufl_warning("Differentiating a conditional with a condition "\
                        "that depends on the differentiation variable."\
                        "This is probably not a good idea!")
        oprime = None # Shouldn't be used anywhere
        return (o, oprime)
    
    def conditional(self, o, c, t, f):
        if isinstance(t[1], Zero) and isinstance(f[1], Zero):
            fi = o.free_indices()
            fid = subdict(o.index_dimensions(), fi)
            return (o, Zero(o.shape(), fi, fid))
        return (o, conditional(c[0], t[1], f[1]))
    
    # --- Derivatives
    
    def spatial_derivative(self, o, f, ii):
        f, fp = f
        ii, iip = ii
        # TODO: Are there any issues with indices here? Not sure, think through it...
        if is_spatially_constant(fp):
            # throw away repeated indices
            fi = tuple(set(f.free_indices()) ^ set(i for i in ii if isinstance(i, Index)))
            fid = f.index_dimensions()
            index_dimensions = dict((i, fid.get(i, self._spatial_dim)) for i in fi)
            oprime = Zero(fp.shape(), fi, index_dimensions)
        else:
            oprime = type(o)(fp, ii)
        return (o, oprime)
    
    def variable_derivative(self, o, f, v):
        f, fp = f
        v, vp = v
        ufl_assert(isinstance(vp, Zero), "TODO: What happens if vp != 0, i.e. v depends the differentiation variable?")
        # TODO: Are there any issues with indices here? Not sure, think through it...
        oprime = type(o)(fp, v)
        return (o, oprime)
    
    def grad(self, o, a):
        a, aprime = a
        if aprime.cell() is None:
            ufl_error("TODO: Shape of gradient is undefined.") # Currently calling expand_compounds before AD to avoid this
            oprime = Zero(TODO)
        else:
            oprime = type(o)(aprime)
        return (o, oprime)


class VariableAD(AD):
    def __init__(self, dim, variable):
        AD.__init__(self, dim)
        self._variable = variable
    
    def variable_derivative(self, o):
        ufl_error("Nested variable derivatives not implemented!")
    
    def variable(self, o):
        if o is self._variable:
            return (o, _1) # FIXME: This assumes variable is scalar!
        else:
            self._variable_cache[o._expression] = o
            x2, xdiff = self.visit(o._expression)
            ufl_assert(o is x2, "Oops.")
            return (o, xdiff)


def compute_diff(expression, var):
    "Differentiate expression w.r.t Variable var."
    ufl_assert(var is None or var.shape() == (), "VariableDerivative w.r.t. nonscalar variable not implemented.")
    dim = expression.cell().dim()
    e, ediff = VariableAD(dim, var).visit(expression)
    return ediff


def compute_variable_derivatives(form):
    "Apply AD to form, expanding all VariableDerivative w.r.t variables."
    cell = form.cell()
    ufl_assert(cell is not None, "Need to know the spatial dimension to compute derivatives.")
    spatial_dim = cell.dim()
    def _compute_diff(expression):
        expression = expand_compounds(expression, spatial_dim)
        return compute_diff(expression, None)
    return transform_integrands(form, _compute_diff)


def propagate_spatial_derivatives(form):
    """Partially apply automatic differentiation to form
    by propagating spatial derivatives to terminal objects."""

    ufl_assert(not extract_type(form, SpatialDerivative), "propagate_spatial_derivatives not implemented")

    cell = form.cell()
    ufl_assert(cell is not None, "Need to know the spatial dimension to compute derivatives.")
    spatial_dim = cell.dim()

    def _compute_diff(expression):
        expression = expand_compounds(expression, spatial_dim)
        # FIXME: Implement!
        return expression

    return transform_integrands(form, _compute_diff)


class FunctionalAD(AD):
    "Apply AFD (Automatic Function Differentiation) to expression."
    def __init__(self, spatial_dim, function, basisfunction):
        AD.__init__(self, spatial_dim)
        
        if isinstance(function, tuple):
            # We got a tuple of functions, handle it as functions
            # over components of a mixed element.
            ufl_assert(all(isinstance(w, Function) for w in function),
                "Expecting a tuple of Functions to differentiate w.r.t.")
            
            elements = [w.element() for w in function]
            element = MixedElement(*elements)
            
            if basisfunction is None:
                basisfunction = BasisFunctions(element)
            else:
                ufl_assert(isinstance(basisfunction, BasisFunction) \
                    and basisfunction == element,
                    "Basis function over wrong element supplied, "\
                    "got %s but expecting %s." % \
                    (repr(basisfunction.element()), repr(element)))
            
            functions = zip(function, basisfunction)
        
        else:
            ufl_assert(isinstance(function, Function),
                "Expecting a Function to differentiate w.r.t.")
            
            if basisfunction is None:
                basisfunction = BasisFunction(function.element())
            functions = [(function, basisfunction)]
        
        self._functions = functions
    
    def function(self, o):
        for (w, wprime) in self._functions:
            if o == w:
                return (w, wprime)
        return (o, Zero(o.shape()))
    
    def variable(self, o):
        dummy, wprime = self.visit(o._expression)
        return (o, wprime)


def compute_form_derivative(form, function, basisfunction):
    "Apply AFD (Automatic Function Differentiation) to Form."
    
    cell = form.cell()
    ufl_assert(cell is not None, "Need to know the spatial dimension to compute derivatives.")
    spatial_dim = cell.dim()
    
    visitor = FunctionalAD(spatial_dim, function, basisfunction)
    
    def _compute_derivative(expression):
        expression = expand_compounds(expression, spatial_dim)
        F, J = visitor.visit(expression)
        return J
    
    return transform_integrands(form, _compute_derivative)

# ================================================================================ OLD CODE ABOVE

def expand_derivatives(form):
    """Expand all derivatives in form, such that the only
    ones left are spatial derivatives applied to terminals."""
    
    cell = form.cell()
    ufl_assert(cell is not None, "Need to know the spatial dimension to compute derivatives.")
    spatial_dim = cell.dim()
    
    def _expand_derivatives(expression):
        expression = expand_compounds(expression, spatial_dim)
        return apply_ad(expression, reverse_ad)
    
    return transform_integrands(form, _expand_derivatives)

