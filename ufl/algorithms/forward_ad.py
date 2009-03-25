"""Forward mode AD implementation."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-19-- 2009-03-25"

from ufl.log import error, warning, debug
from ufl.assertions import ufl_assert
from ufl.common import unzip, subdict, lstr
from ufl.indexutils import unique_indices

# All classes:
from ufl.terminal import Terminal, Tuple
from ufl.constantvalue import Zero, IntValue, Identity, is_true_ufl_scalar
from ufl.variable import Variable
from ufl.function import ConstantBase
from ufl.indexing import MultiIndex, Indexed, Index, indices
from ufl.indexsum import IndexSum
from ufl.tensors import ListTensor, ComponentTensor, as_tensor, as_scalar
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.differentiation import Derivative, FunctionDerivative, SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional

# Lists of all Expr classes
#from ufl.classes import ufl_classes, terminal_classes, nonterminal_classes
from ufl.classes import terminal_classes
from ufl.operators import dot, inner, outer, lt, eq, conditional, sign
from ufl.operators import sqrt, exp, ln, cos, sin
from ufl.algorithms.traversal import iter_expressions
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.transformations import expand_compounds, Transformer, transform, transform_integrands

from ufl.differentiation import is_spatially_constant

class ForwardAD(Transformer):
    def __init__(self, spatial_dim, var_shape, var_free_indices, var_index_dimensions):
        Transformer.__init__(self)
        ufl_assert(all(isinstance(i, Index) for i in var_free_indices), \
            "Expecting Index objects.")
        ufl_assert(all(isinstance(i, Index) for i in var_index_dimensions.keys()), \
            "Expecting Index objects.")
        self._spatial_dim = spatial_dim
        self._var_shape = var_shape
        self._var_free_indices = var_free_indices
        self._var_index_dimensions = dict(var_index_dimensions)
        #if self._var_free_indices:
        #    warning("TODO: Free indices in differentiation variable may be buggy!")
    
    def _make_zero_diff(self, o):
        # Define a zero with the right indices
        # (kind of cumbersome this... any simpler way?)
        sh = o.shape() + self._var_shape
        fi = o.free_indices()
        idims = dict(o.index_dimensions())
        if self._var_free_indices:
            # Currently assuming only one free variable index
            i, = self._var_free_indices 
            if i not in idims:
                fi = unique_indices(fi + (i,))
                idims[i] = self._var_index_dimensions[i]
        fp = Zero(sh, fi, idims)
        return fp   

    def _make_ones_diff(self, o):
        # Define a scalar value with the right indices
        # (kind of cumbersome this... any simpler way?)
        sh = o.shape() + self._var_shape
        fi = o.free_indices()
        idims = dict(o.index_dimensions())
        if self._var_free_indices:
            # Currently assuming only one free variable index
            i, = self._var_free_indices
            if i not in idims:
                fi = unique_indices(fi + (i,))
                idims[i] = self._var_index_dimensions[i]
        fp = IntValue(1, sh, fi, idims)
        return fp
    
    def _visit(self, o):
        "Debugging hook, enable this by renaming to 'visit'."
        r = Transformer.visit(self, o)
        f, df = r
        if not f is o:
            debug("In ForwardAD.visit, didn't get back o:")
            debug("  o:  %s" % str(o))
            debug("  f:  %s" % str(f))
            debug("  df: %s" % str(df))
        fi_diff = set(f.free_indices()) ^ set(df.free_indices())
        if fi_diff:
            debug("In ForwardAD.visit, got free indices diff:")
            debug("  o:  %s" % str(o))
            debug("  f:  %s" % str(f))
            debug("  df: %s" % str(df))
            debug("  f.fi():  %s" % lstr(f.free_indices()))
            debug("  df.fi(): %s" % lstr(df.free_indices()))
            debug("  fi_diff: %s" % str(fi_diff))
        return r
    
    # --- Default rules
    
    def expr(self, o):
        error("Missing ForwardAD handler for type %s" % str(type(o)))
    
    def terminal(self, o):
        """Terminal objects are assumed independent of the differentiation
        variable by default, and simply 'lifted' to the pair (o, 0).
        Depending on the context, override this with custom rules for
        non-zero derivatives."""
        fp = self._make_zero_diff(o)
        return (o, fp)
    
    def variable(self, o):
        """Variable objects are just 'labels', so by default the derivative
        of a variable is the derivative of its referenced expression."""
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        r = self._variable_cache.get(l) # cache contains (v, vp) tuple
        if r is not None:
            return r
        
        # Visit the expression our variable represents
        e2, vp = self.visit(e)

        # Recreate Variable (with same label) only if necessary
        v = self.reuse_if_possible(o, e2, l)

        # Cache and return (v, vp) tuple
        r = (v, vp)
        self._variable_cache[l] = r
        return r
    
    # --- Indexing and component handling
    
    def multi_index(self, o):
        return (o, None) # oprime here should never be used, this might even not be called?
    
    def indexed(self, o):
        A, jj = o.operands()
        A2, Ap = self.visit(A)
        o = self.reuse_if_possible(o, A2, jj)
        
        if isinstance(Ap, Zero):
            op = self._make_zero_diff(o)
        else:
            r = Ap.rank() - len(jj)
            ii = indices(r)
            op = Indexed(Ap, jj + ii)
            if ii:
                op = as_tensor(op, ii)
        return (o, op)
    
    def list_tensor(self, o, *ops):
        ops, dops = unzip(ops)
        o = self.reuse_if_possible(o, *ops)
        op = ListTensor(*dops)
        return (o, op)
    
    def component_tensor(self, o):
        A, ii = o.operands()
        A, Ap = self.visit(A)
        o = self.reuse_if_possible(o, A, ii)
        
        if isinstance(Ap, Zero):
            op = self._make_zero_diff(o)
        else:
            Ap, jj = as_scalar(Ap)
            op = ComponentTensor(Ap, ii + jj)
        return (o, op)
    
    # --- Algebra operators
    
    def index_sum(self, o):
        A, i = o.operands()
        A2, Ap = self.visit(A)
        o = self.reuse_if_possible(o, A2, i)
        op = IndexSum(Ap, i)
        return (o, op)
    
    def sum(self, o, *ops):
        ops, opsp = unzip(ops)
        o2 = self.reuse_if_possible(o, *ops)
        op = sum(opsp[1:], opsp[0])
        return (o2, op)
    
    def product(self, o, *ops):
        # Start with a zero with the right shape and indices
        fp = self._make_zero_diff(o)
        # Get operands and their derivatives
        ops2, dops2 = unzip(ops)
        o = self.reuse_if_possible(o, *ops2)
        for (i, op) in enumerate(ops):
            # Get scalar representation of differentiated value of operand i
            dop = dops2[i]
            dop, ii = as_scalar(dop)
            # Replace operand i with its differentiated value in product
            fpoperands = ops2[:i] + [dop] + ops2[i+1:]
            p = Product(*fpoperands)
            # Wrap product in tensor again
            if ii:
                p = as_tensor(p, ii)
            # Accumulate terms
            fp += p
        return (o, fp)
    
    def division(self, o, a, b):
        f, fp = a
        g, gp = b
        o = self.reuse_if_possible(o, f, g)
        
        ufl_assert(is_true_ufl_scalar(g), "Not expecting nonscalar denominator")
        
        #do_df = 1/g
        #do_dg = -h/g
        #op = do_df*fp + do_df*gp
        #op = (fp - o*gp) / g
        
        # Get o and gp as scalars, multiply, then wrap as a tensor again
        so, oi = as_scalar(o)
        sgp, gi = as_scalar(gp)
        o_gp = so*sgp
        if oi or gi:
            o_gp = as_tensor(o_gp, oi + gi)
        op = (fp - o_gp) / g
        return (o, op)
    
    def power(self, o, a, b):
        f, fp = a
        g, gp = b
        if not is_true_ufl_scalar(f):
            print ":"*80
            print "f =", str(f)
            print "g =", str(g)
            print ":"*80
        ufl_assert(is_true_ufl_scalar(f), "Expecting scalar expression f in f**g.")
        ufl_assert(is_true_ufl_scalar(g), "Expecting scalar expression g in f**g.")
        
        # General case: o = f(x)**g(x)
        
        #do_df = g * f**(g-1)
        #do_dg = ln(f) * f**g
        #op = do_df*fp + do_dg*gp
        
        #do_df = o * g / f # f**g * g / f
        #do_dg = ln(f) * o
        #op = do_df*fp + do_dg*gp
        
        # Got two possible alternatives here:
        if False:
            # Pulling o out gives:
            op = o*(fp*g/f + ln(f)*gp)
            # This produces expressions like (1/w)*w**5 instead of w**4
            # If we do this, we reuse o
            o = self.reuse_if_possible(o, f, g)
        else:
            # Rewriting o as f*f**(g-1) we can do:
            f_g_m1 = f**(g-1)
            op = f_g_m1*(fp*g + f*ln(f)*gp)
            # In this case we can rewrite o using new subexpression
            o = f*f_g_m1
        
        return (o, op)
    
    def abs(self, o, a):
        f, fprime = a
        o = self.reuse_if_possible(o, f)
        oprime = conditional(eq(f, 0),
                             0,
                             Product(sign(f), fprime)) #conditional(lt(f, 0), -fprime, fprime))
        return (o, oprime)
    
    # --- Mathfunctions
    
    def math_function(self, o, a):
        error("Unknown math function.")
    
    def sqrt(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp / (2*o)
        return (o, op)
    
    def exp(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*o
        return (o, op)
    
    def ln(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        ufl_assert(not isinstance(f, Zero), "Division by zero.")
        return (o, fp/f)
    
    def cos(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = -fp*sin(f)
        return (o, op)
    
    def sin(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*cos(f)
        return (o, op)
    
    # --- Restrictions
    
    def positive_restricted(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        return (o, fp('+')) # TODO: Assuming here that restriction and differentiation commutes. Is this correct?
    
    def negative_restricted(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        return (o, fp('-')) # TODO: Assuming here that restriction and differentiation commutes. Is this correct?
    
    # --- Conditionals
    
    def condition(self, o, l, r):
        o = self.reuse_if_possible(o, l[0], r[0])
        if any(not isinstance(op[1], Zero) for op in (l, r)):
            warning("Differentiating a conditional with a condition "\
                        "that depends on the differentiation variable."\
                        "This is probably not a good idea!")
        oprime = None # Shouldn't be used anywhere
        return (o, oprime)
    
    def conditional(self, o, c, t, f):
        o = self.reuse_if_possible(o, c[0], t[0], f[0])
        if isinstance(t[1], Zero) and isinstance(f[1], Zero):
            fi = o.free_indices()
            fid = subdict(o.index_dimensions(), fi)
            op = Zero(o.shape(), fi, fid)
        else:
            op = conditional(c[0], t[1], f[1])
        return (o, op)
    
    # --- Other derivatives
    
    def derivative(self, o):
        error("This should never occur.")
    
    def _spatial_derivative(self, o):
        # If everything else works as it should, this should now 
        # be treated as a "terminal" in the context of AD,
        # i.e. the differentiation this represents has already
        # been applied. TODO: Document the reason for this well!
        
        # TODO: Although differentiation commutes, can we get repeated index issues here?
        f, i = o.operands()
        f, fp = self.visit(f)
        o = self.reuse_if_possible(o, f, i)
        op = SpatialDerivative(fp, i) # FIXME
        return (o, op)
    
    def spatial_derivative(self, o): # FIXME: Fix me!
        # If we hit this type, it has already been propagated
        # to a terminal, so we can simply apply our derivative
        # to its operand since differentiation commutes. Right?
        f, ii = o.operands()
        f, fp = self.visit(f)
        o = self.reuse_if_possible(o, f, ii)
        
        # TODO: Are there any issues with indices here? Not sure, think through it...
        if is_spatially_constant(fp):
            fi = f.free_indices()
            idims = f.index_dimensions()
            
            # throw away repeated indices # TODO: Think these can be removed now
            #fi = tuple(set(fi) ^ set(i for i in ii if isinstance(i, Index)))
            #idims= dict((i, idims.get(i, self._spatial_dim)) for i in fi)
            
            oprime = Zero(fp.shape(), fi, idims)
        else:
            oprime = SpatialDerivative(fp, ii)
        return (o, oprime)

class SpatialAD(ForwardAD):
    def __init__(self, spatial_dim, index):
        # FIXME: Iron out this, decide where to use MultiIndex and Index properly
        if isinstance(index, MultiIndex):
            index, = index
        if isinstance(index, Index):
            vfi = (index,)
            vid = { index: spatial_dim }
        else:
            vfi = ()
            vid = {}
        ForwardAD.__init__(self, spatial_dim, var_shape=(), var_free_indices=vfi, var_index_dimensions=vid)
        self._index = index
    
    def spatial_coordinate(self, o):
        # TODO: Need to define dx_i/dx_j = delta_ij?
        error("Not implemented!")
        I = Identity(self._spatial_dim)
        oprime = I[:, self._index] # TODO: Is this right?
        return (o, oprime)
    
    def basis_function(self, o):
        # FIXME: Using this index in here may collide with the same index on the outside!
        # FIXME: Can this give recursion in apply_ad?
        #oprime = o.dx(self._index)
        oprime = SpatialDerivative(o, self._index)
        return (o, oprime)
    
    def function(self, o):
        #oprime = o.dx(self._index)
        oprime = SpatialDerivative(o, self._index)
        return (o, oprime)
    
    constant = ForwardAD.terminal # returns zero
    
    #def facet_normal(self, o):
    #    pass # TODO: With higher order cells the facet normal isn't constant anymore

class VariableAD(ForwardAD):
    def __init__(self, spatial_dim, var):
        ForwardAD.__init__(self, spatial_dim, var_shape=var.shape(), var_free_indices=var.free_indices(), var_index_dimensions=var.index_dimensions())
        self._variable = var
    
    def variable(self, o): # XXX: This is another example
        # Check cache
        e, l = o.operands()
        c = self._variable_cache.get(l)
        
        if c is not None:
            return c
        
        if o.label() == self._variable.label():
            # dv/dv = 1
            op = self._make_ones_diff(o)
        else:
            # differentiate expression behind variable
            e2, ep = self.visit(e)
            op = ep
            if not e2 == e:
                o = Variable(e2, l)
        # return variable and derivative of its expression
        c = (o, op)
        self._variable_cache[l] = c
        return c

class FunctionAD(ForwardAD):
    "Apply AFD (Automatic Function Differentiation) to expression."
    def __init__(self, spatial_dim, functions, basis_functions):
        ForwardAD.__init__(self, spatial_dim, var_shape=(), var_free_indices=(), var_index_dimensions={})
        self._functions = zip(functions, basis_functions)
        self._w = functions
        self._v = basis_functions
        ufl_assert(isinstance(self._w, Tuple), "Eh?")
        ufl_assert(isinstance(self._v, Tuple), "Eh?")
        # Define dw/dw := v (what we really mean by d/dw is d/dw_j where w = sum_j w_j phi_j, and what we really mean by v is phi_j for any j)
    
    def function(self, o):
        debug("In FunctionAD.function:")
        debug("o = %s" % o)
        debug("self._w = %s" % self._w)
        debug("self._v = %s" % self._v)
        for (w, v) in zip(self._w, self._v): #self._functions:
            if o == w:
                return (w, v)
        return (o, Zero(o.shape()))
    
    def variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        c = self._variable_cache.get(l)
        if c is not None:
            return c
        
        # Visit the expression our variable represents
        e2, ep = self.visit(e)
        op = ep

        # If the expression is not the same, reconstruct Variable object
        o = self.reuse_if_possible(o, e2, l)

        # Recreate Variable (with same label) and cache it
        c = (o, op)
        self._variable_cache[l] = c
        return c

def compute_spatial_forward_ad(expr, dim):
    f, v = expr.operands()
    alg = SpatialAD(dim, v)
    e, ediff = alg.visit(f)
    return ediff

def compute_variable_forward_ad(expr, dim):
    f, v = expr.operands()
    alg = VariableAD(dim, v)
    e, ediff = alg.visit(f)
    return ediff

def compute_function_forward_ad(expr, dim):
    f, w, v = expr.operands()
    alg = FunctionAD(dim, w, v)
    e, ediff = alg.visit(f)
    return ediff

def forward_ad(expr, dim):
    """Assuming expr is a derivative and contains no other
    unresolved derivatives, apply forward mode AD and
    return the computed derivative."""
    if isinstance(expr, SpatialDerivative):
        result = compute_spatial_forward_ad(expr, dim)
    elif isinstance(expr, VariableDerivative):
        result = compute_variable_forward_ad(expr, dim)
    elif isinstance(expr, FunctionDerivative):
        result = compute_function_forward_ad(expr, dim)
    else:
        error("This shouldn't happen: expr is %s" % repr(expr))
    return result


#
# TODO: We could expand only the compound objects that have no rule
#       before differentiating, to allow the AD to work on a coarser graph
#       (Missing rules for: Cross, Determinant, Cofactor)
#
class UnusedADRules(ForwardAD):
    
    def _variable_derivative(self, o, f, v):
        f, fp = f
        v, vp = v
        ufl_assert(isinstance(vp, Zero), "TODO: What happens if vp != 0, i.e. v depends the differentiation variable?")
        # Are there any issues with indices here? Not sure, think through it...
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
            error("TODO: Shape of gradient is undefined.") # Currently calling expand_compounds before AD to avoid this
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
        error("Derivative of cross product not implemented, apply expand_compounds before AD.")
        u, up = a
        v, vp = b
        #oprime = ...
        return (o, oprime)
    
    def determinant(self, o, a):
        error("Derivative of determinant not implemented, apply expand_compounds before AD.")
        A, Ap = a
        #oprime = ...
        return (o, oprime)
    
    def cofactor(self, o, a):
        error("Derivative of cofactor not implemented, apply expand_compounds before AD.")
        A, Ap = a
        #cofacA_prime = detA_prime*Ainv + detA*Ainv_prime
        #oprime = ...
        return (o, oprime)
    
    def inverse(self, o, a):
        """Derivation:
        0 = d/dx [Ainv*A] = Ainv' * A + Ainv * A'
        Ainv' * A = - Ainv * A'
        Ainv' = - Ainv * A' * Ainv
        """
        A, Ap = a
        return (o, -o*Ap*o) # Any potential index problems here?
    
