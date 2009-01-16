"""Forward mode AD implementation."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-19-- 2009-01-16"

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
from ufl.classes import Terminal, Expr, Derivative, Tuple, SpatialDerivative, VariableDerivative, FunctionDerivative

# Lists of all Expr classes
#from ufl.classes import ufl_classes, terminal_classes, nonterminal_classes
from ufl.classes import terminal_classes
from ufl.operators import dot, inner, outer, lt, eq, conditional
from ufl.operators import sqrt, exp, ln, cos, sin
from ufl.algorithms.traversal import iter_expressions
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.transformations import expand_compounds, Transformer, transform, transform_integrands

#
# TODO: Missing rule for: f(x)**g(x))
# TODO: We could expand only the compound objects that have no rule
#       before differentiating, to make the AD work on a coarser graph
#       (Missing rules for: Cross, Determinant, Cofactor)
#

def is_spatially_constant(o):
    return (isinstance(o, Terminal) and o.cell() is None) or isinstance(o, Constant)

_0 = Zero()
_1 = IntValue(1)

class AD(Transformer):
    def __init__(self, spatial_dim):
        Transformer.__init__(self)
        self._spatial_dim = spatial_dim
    
    def visit(*args):
        result = Transformer.visit(*args)
        # FIXME: Inspect results here for debugging
        return result
    
    # --- Default rules
    
    def expr(self, o):
        ufl_error("Missing AD handler for type %s" % str(type(o)))
    
    def terminal(self, o):
        """Terminal objects are assumed independent of the differentiation
        variable by default, and simply 'lifted' to the pair (o, 0).
        Depending on the context, override this with custom rules for
        non-zero derivatives."""
        return (o, Zero(o.shape()))
    
    def variable(self, o):
        """Variable objects are just 'labels', so by default the derivative
        of a variable is the derivative of its referenced expression."""
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        r = self._variable_cache.get(l) # cache contains (v, vp) tuple
        if r is None:
            # Visit the expression our variable represents
            e2, vp = self.visit(e)
            # Recreate Variable (with same label) only if necessary
            if e is e2:
                v = o
            else:
                v = Variable(e2, l)
            # Cache and return (v, vp) tuple
            r = (v, vp)
            self._variable_cache[l] = r
        return r
    
    # --- Indexing and component handling
    
    def multi_index(self, o):
        return (o, None) # oprime here should never be used
    
    def indexed(self, o, A, ii):
        return (o, A[1][ii[0]])
    
    def list_tensor(self, o, *ops):
        opprimes = [op[1] for op in ops]
        return (o, ListTensor(*opprimes))
    
    def component_tensor(self, o, A, ii):
        A, Ap = A
        if isinstance(Ap, Zero):
            fi = o.free_indices()
            fid = subdict(o.index_dimensions(), fi)
            return (o, Zero(o.shape(), fi, fid))
        return (o, ComponentTensor(Ap, ii[0]) )
    
    # --- Algebra operators
    
    def sum(self, o, *ops):
        return (sum((op[0] for op in ops[1:]), ops[0][0]),
                sum((op[1] for op in ops[1:]), ops[0][1]))
    
    def _product(self, o, *ops):
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
    
    def product(self, o, *ops):
        fi = o.free_indices()
        fid = subdict(o.index_dimensions(), fi)
        fp = Zero(o.shape(), fi, fid)
        ops2, dops2 = unzip(ops)
        
        for (i, op) in enumerate(ops):
            # replace operand i with its differentiated value 
            fpoperands = ops2[:i] + [dops2[i]] + ops2[i+1:]
            # simplify by ignoring ones
            fpoperands = [fpop for fpop in fpoperands if not fpop == 1]
            # simplify if there are zeros in the product
            if not any(isinstance(fpop, Zero) for fpop in fpoperands):
                fp += product(fpoperands) # FIXME: fp and product(fpoperands) may have different free indices, causing this to fail!
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
    
    # --- Mathfunctions
    
    def math_function(self, o, a):
        f, fp = a
        return (o, 0.5*fp/sqrt(f))
    
    def sqrt(self, o, a):
        f, fp = a
        return (o, 0.5*fp/sqrt(f))
    
    def exp(self, o, a):
        f, fp = a
        return (o, fp*exp(f))
    
    def ln(self, o, a):
        f, fp = a
        ufl_assert(not isinstance(f, Zero), "Division by zero.")
        return (o, fp/f)
    
    def cos(self, o, a):
        f, fp = a
        return (o, -fp*sin(f))
    
    def sin(self, o, a):
        f, fp = a
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
    
    # --- Other derivatives
    
    def derivative(self, o):
        ufl_error("This should never occur.")
    
    def _spatial_derivative(self, o):
        # If everything else works as it should, this should now 
        # be treated as a "terminal" in the context of AD,
        # i.e. the differentiation this represents has already
        # been applied. TODO: Document the reason for this well!
        
        # TODO: Although differentiation commutes, can we get repeated index issues here?
        f, i = o.operands()
        f, fp = self.visit(f)
        op = o._uflid(fp, i) # FIXME
        return (o, op)
    
    def spatial_derivative(self, o): # FIXME: Fix me!
        # If we hit this type, it has already been propagated
        # to a terminal, so we can simply apply our derivative
        # to its operand since differentiation commutes. Right?
        f, ii = o.operands()
        f, fp = self.visit(f)
        
        # TODO: Are there any issues with indices here? Not sure, think through it...
        if is_spatially_constant(fp):
            # throw away repeated indices
            fi = tuple(set(f.free_indices()) ^ set(i for i in ii if isinstance(i, Index)))
            fid = f.index_dimensions()
            index_dimensions = dict((i, fid.get(i, self._spatial_dim)) for i in fi)
            oprime = Zero(fp.shape(), fi, index_dimensions)
        else:
            oprime = o._uflid(fp, ii)
        return (o, oprime)

class SpatialAD(AD):
    def __init__(self, dim, index):
        AD.__init__(self, dim)
        self._index = index
    
    def spatial_coordinate(self, o):
        # TODO: Need to define dx_i/dx_j = delta_ij?
        ufl_error("Not implemented!")
        I = Identity(self._spatial_dim)
        oprime = I[:, self._index] # TODO: Is this right?
        return (o, oprime)
    
    def basis_function(self, o):
        # FIXME: Using this index in here may collide with the same index on the outside!
        # FIXME: Can this give recursion in apply_ad?
        oprime = o.dx(self._index) # TODO: Add derivatives field to BasisFunction?
        return (o, oprime)
    
    def function(self, o):
        oprime = o.dx(self._index) # TODO: Add derivatives field to Function?
        return (o, oprime)
    
    constant = AD.terminal # returns zero
    
    #def facet_normal(self, o):
    #    pass # TODO: With higher order cells the facet normal isn't constant anymore

class VariableAD(AD):
    def __init__(self, dim, variable):
        AD.__init__(self, dim)
        self._variable = variable
    
    def variable(self, o):
        if o is self._variable:
            return (o, _1) # FIXME: This assumes variable is scalar!
        else:
            self._variable_cache[o._expression] = o
            x2, xdiff = self.visit(o._expression)
            ufl_assert(o is x2, "How did this happen?")
            return (o, xdiff)

class FunctionAD(AD):
    "Apply AFD (Automatic Function Differentiation) to expression."
    def __init__(self, spatial_dim, functions, basisfunctions):
        AD.__init__(self, spatial_dim)
        self._functions = zip(functions, basisfunctions)
        self._w = functions
        self._v = basisfunctions
        ufl_assert(isinstance(self._w, Tuple), "Eh?")
        ufl_assert(isinstance(self._v, Tuple), "Eh?")
        # Define dw/dw := v (what we really mean by d/dw is d/dw_j where w = sum_j w_j phi_j, and what we really mean by v is phi_j for any j)
    
    def function(self, o):
        print "In FunctionAD.function:"
        print "o = ", o
        print "self._w = ", self._w
        print "self._v = ", self._v
        for (w, v) in zip(self._w, self._v): #self._functions:
            if o == w:
                return (w, v)
        return (o, Zero(o.shape()))

    def variable(self, o):
        dummy, wprime = self.visit(o._expression)
        return (o, wprime)

def compute_spatial_forward_ad(expr):
    cell = expr.cell()
    ufl_assert(cell is not None, "Need spatial dimension to compute derivatives, can't get that from %s." % repr(expr))
    dim = cell.dim()
    expr = expand_compounds(expr, dim)
    f, v = expr.operands()
    e, ediff = SpatialAD(dim, v).visit(f)
    return ediff

def compute_variable_forward_ad(expr):
    cell = expr.cell()
    ufl_assert(cell is not None, "Need spatial dimension to compute derivatives, can't get that from %s." % repr(expr))
    dim = cell.dim()
    expr = expand_compounds(expr, dim)
    f, v = expr.operands()
    ufl_assert(v.shape() == (), "compute_variable_forward_ad with nonscalar Variable not implemented.")
    e, ediff = VariableAD(dim, v).visit(f)
    return ediff

def compute_function_forward_ad(expr):
    cell = expr.cell()
    ufl_assert(cell is not None, "Need spatial dimension to compute derivatives, can't get that from %s." % repr(expr))
    dim = cell.dim()
    expr = expand_compounds(expr, dim)
    f, w, v = expr.operands()
    e, ediff = FunctionAD(dim, w, v).visit(f)
    return ediff

def forward_ad(expr):
    """Assuming expr is a derivative and contains no other
    unresolved derivatives, apply forward mode AD and
    return the computed derivative."""
    if isinstance(expr, SpatialDerivative):
        result = compute_spatial_forward_ad(expr)
    elif isinstance(expr, VariableDerivative):
        result = compute_variable_forward_ad(expr)
    elif isinstance(expr, FunctionDerivative):
        result = compute_function_forward_ad(expr)
    else:
        ufl_warning("How did this happen? expr is %s" % repr(expr))
        result = expr
    return result


class UnusedADRules(AD):
    
    def _variable_derivative(self, o, f, v):
        f, fp = f
        v, vp = v
        ufl_assert(isinstance(vp, Zero), "TODO: What happens if vp != 0, i.e. v depends the differentiation variable?")
        # TODO: Are there any issues with indices here? Not sure, think through it...
        oprime = type(o)(fp, v)
        return (o, oprime)
    
    # --- Compound operators
    
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
    def grad(self, o, a):
        a, aprime = a
        if aprime.cell() is None:
            ufl_error("TODO: Shape of gradient is undefined.") # Currently calling expand_compounds before AD to avoid this
            oprime = Zero(TODO)
        else:
            oprime = type(o)(aprime)
        return (o, oprime)
    
    def outer(self, o, a, b):
        a, ap = a
        b, bp = b
        return (o, outer(ap, b) + outer(a, bp))
    
    def inner(self, o, a, b):
        a, ap = a
        b, bp = b
        return (o, inner(ap, b) + inner(a, bp))
    
    def dot(self, o, a, b):
        a, ap = a
        b, bp = b
        return (o, dot(ap, b) + dot(a, bp))
    
    def cross(self, o, a, b):
        ufl_error("Derivative of cross product not implemented, apply expand_compounds before AD.")
        u, up = a
        v, vp = b
        oprime = None # TODO
        return (o, oprime)
    
    def determinant(self, o, a):
        ufl_error("Derivative of determinant not implemented, apply expand_compounds before AD.")
        A, Ap = a
        oprime = None # TODO
        return (o, oprime)
    
    def cofactor(self, o, a):
        ufl_error("Derivative of cofactor not implemented, apply expand_compounds before AD.")
        A, Ap = a
        #cofacA_prime = detA_prime*Ainv + detA*Ainv_prime
        oprime = None # TODO
        return (o, oprime)
    
    def inverse(self, o, a):
        """Derivation:
        0 = d/dx [Ainv*A] = Ainv' * A + Ainv * A'
        Ainv' * A = - Ainv * A'
        Ainv' = - Ainv * A' * Ainv
        """
        A, Ap = a
        return (o, -o*Ap*o)
    
