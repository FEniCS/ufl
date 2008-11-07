"""This module defines automatic differentiation utilities."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-19-- 2008-11-06"

from ..output import ufl_assert, ufl_error, ufl_warning
from ..common import product, unzip, UFLTypeDefaultDict, domain2dim

# All classes:
from ..base import Expr, Terminal
from ..zero import Zero
from ..scalar import FloatValue, IntValue
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunction import BasisFunction, BasisFunctions
from ..function import Function, Constant
from ..indexing import MultiIndex, Indexed, Index
from ..tensors import ListTensor, ComponentTensor
from ..algebra import Sum, Product, Division, Power, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional

from ..classes import ScalarValue, Zero, Identity, Constant, VectorConstant, TensorConstant

# Lists of all Expr classes
#from ..classes import ufl_classes, terminal_classes, nonterminal_classes
from ..classes import terminal_classes
from ..operators import dot, inner, outer, lt, eq, conditional
from ..operators import sqrt, exp, ln, cos, sin
from .traversal import iter_expressions
from .analysis import extract_type
from .transformations import transform, transform_integrands, expand_compounds

# FIXME: Need algorithm to apply AD to all kinds of derivatives!
#        In particular, SpatialDerivative, VariableDerivative and functional derivative.

# FIXME: Need some cache structures and callback to custum diff routine to implement diff with variable
# - Check for diff of variable in some kind of cache
# - If not found, apply diff to variable expression 
# - Add variable for differentated expression to cache

# FIXME: Missing rules for:
# Cross, Determinant, Cofactor, f(x)**g(x)
# FIXME: Could apply as_basic to Compound objects with no rule before differentiating

spatially_constant_types = (ScalarValue, Zero, Identity, Constant, VectorConstant, TensorConstant) # FacetNormal: not for higher order geometry!

def diff_handlers():
    """This function constructs a default handler dict for 
    compute_derivative. Nonterminal objects are reused if possible."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in diff_handlers. Add to classes.py." % type(x))
    d = UFLTypeDefaultDict(not_implemented)
    
    _zero = Zero()
    
    # Terminal objects are assumed independent of the differentiation
    # variable by default, and simply 'lifted' to the pair (x, 0).
    # Depending on the context, override this with custom rules for
    # non-zero derivatives.
    def lift(x):
        return (x, Zero(x.shape()))
    for c in terminal_classes:
        d[c] = lift
    
    # This should work for all single argument operators that commute with d/dw:
    def diff_commute(x, *ops):
        ufl_assert(len(ops) == 1, "Logic breach in diff_commute, len(ops) = %d." % len(ops))
        oprime = ops[0][1]
        return (x, type(x)(oprime))
    
    def diff_variable(x, *ops):
        ufl_error("How to handle derivative of variable depends on context. You must supply a customized rule!")
    d[Variable] = diff_variable

    # These differentiation rules for nonterminal objects should probably never need to be overridden:
    def diff_multi_index(x, *ops):
        return (x, None) # x' here should never be used
    d[MultiIndex] = diff_multi_index
    
    def diff_indexed(x, *ops):
        A, i = ops
        return (x, A[1][i[0]])
    d[Indexed] = diff_indexed
    
    def diff_listtensor(x, *ops):
        ops1 = [o[1] for o in ops]
        return (x, ListTensor(*ops1))
    d[ListTensor] = diff_listtensor
    
    def diff_tensor(x, *ops):
        A, i = ops
        if isinstance(A[1], Zero):
            return (x, Zero(x.shape()))
        return (x, ComponentTensor(A[1], i[0]) )
    d[ComponentTensor] = diff_tensor
    
    def diff_sum(x, *ops):
        return (sum((o[0] for o in ops[1:]), ops[0][0]),
                sum((o[1] for o in ops[1:]), ops[0][1]))
    d[Sum] = diff_sum
    
    def diff_product(x, *ops):
        fp = Zero(x.shape())
        ops0, ops1 = unzip(ops)
        for (i,o) in enumerate(ops):
            # replace operand i with its differentiated value 
            fpoperands = ops0[:i] + [ops1[i]] + ops0[i+1:]
            # simplify by ignoring ones
            fpoperands = [o for o in fpoperands if not o == 1]
            # simplify if there are zeros in the product
            if not any(isinstance(o, Zero) for o in fpoperands):
                fp += product(fpoperands)
        return (x, fp)
    d[Product] = diff_product
    
    def diff_division(x, *ops):
        f, fp = ops[0]
        g, gp = ops[1]
        return (x, (fp-f*gp/g)/g)
        #return (x, (fp*g-f*gp)/g**2)
    d[Division] = diff_division
    
    def diff_power(x, *ops):
        f, fp = ops[0]
        g, gp = ops[1]
        ufl_assert(not (f.shape() or g.shape()), "Expecting scalar expressions f,g in f**g.")
        # x = f**g
        f_const = isinstance(fp, Zero)
        g_const = isinstance(gp, Zero)
        # Case: x = const ** const = const
        if f_const and g_const:
            return (x, _zero)
        # Case: x = f(x) ** const
        if g_const:
            # x' = g f'(x) f(x)**(g-1)
            if isinstance(g, Zero) or isinstance(f, Zero) or f_const:
                return (x, _zero)
            return (x, g*fp*f**(g-1.0))
        # Case: x = f ** g(x)
        if isinstance(fp, Zero):
            return (x, gp*ln(f)*x)
        ufl_error("diff_power not implemented for case d/dx [ f(x)**g(x) ].")
        return (x, FIXME)
    d[Power] = diff_power
    
    def diff_abs(x, *ops):
        f, fprime = ops[0]
        xprime = conditional(eq(f, 0),
                             0,
                             conditional(lt(f, 0), -fprime, fprime))
        return (x, xprime)
    d[Abs] = diff_abs
    
    d[Transposed] = diff_commute
    
    def diff_outer(x, *ops):
        a, ap = ops[0]
        b, bp = ops[1]
        return (x, outer(ap, b) + outer(a, bp))
    d[Outer] = diff_outer
    
    def diff_inner(x, *ops):
        a, ap = ops[0]
        b, bp = ops[1]
        return (x, inner(ap, b) + inner(a, bp))
    d[Inner] = diff_inner
    
    def diff_dot(x, *ops):
        a, ap = ops[0]
        b, bp = ops[1]
        return (x, dot(ap, b) + dot(a, bp))
    d[Dot] = diff_dot
    
    def diff_cross(x, *ops):
        u, up = ops[0]
        v, vp = ops[1]
        ufl_error("diff_cross not implemented, apply expand_compounds before AD.")
        return (x, FIXME) # COMPOUND
    d[Cross] = diff_cross
    
    d[Trace] = diff_commute
    
    def diff_determinant(x, *ops):
        A, Ap = ops[0]
        ufl_error("diff_determinant not implemented, apply expand_compounds before AD.")
        return (x, FIXME) # COMPOUND
    d[Determinant] = diff_determinant
    
    # Derivation:
    # 0 = d/dx [Ainv*A] = Ainv' * A + Ainv * A'
    # Ainv' * A = - Ainv * A'
    # Ainv' = - Ainv * A' * Ainv
    def diff_inverse(Ainv, *ops):
        A, Ap = ops[0]
        return (Ainv, -Ainv*Ap*Ainv)
    d[Inverse] = diff_inverse
    
    d[Deviatoric] = diff_commute
    
    def diff_cofactor(x, *ops):
        A, Ap = ops[0]
        ufl_error("diff_cofactor not implemented, apply expand_compounds before AD.")
        #cofacA_prime = detA_prime*Ainv + detA*Ainv_prime
        return (x, FIXME) # COMPOUND
    d[Cofactor] = diff_cofactor

    # Mathfunctions:
    def diff_sqrt(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, Zero): return (x, _zero)
        return (x, 0.5*fp/sqrt(f))
    d[Sqrt] = diff_sqrt
    
    def diff_exp(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, Zero):
            ufl_assert(not isinstance(fp*exp(f), Zero), "If this fails, we can remove this code.")
            return (x, _zero)# TODO: don't need this anymore?
        return (x, fp*exp(f))
    d[Exp] = diff_exp
    
    def diff_ln(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, Zero): return (x, _zero)
        ufl_assert(not isinstance(f, Zero), "Division by zero.")
        return (x, fp/f)
    d[Ln] = diff_ln
    
    def diff_cos(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, Zero): return (x, _zero)
        return (x, -fp*sin(f))
    d[Cos] = diff_cos
    
    def diff_sin(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, Zero): return (x, _zero)
        return (x, fp*cos(f))
    d[Sin] = diff_sin

    # Restrictions
    def diff_positiverestricted(x, *ops):
        f, fp = ops[0]
        return (x, fp('+')) # TODO: What is d(v+)/dw ? Assuming here that restriction and differentiation commutes.
    d[PositiveRestricted] = diff_positiverestricted

    def diff_negativerestricted(x, *ops):
        f, fp = ops[0]
        return (x, fp('-')) # TODO: What is d(v-)/dw ? Assuming here that restriction and differentiation commutes.
    d[NegativeRestricted] = diff_negativerestricted
    
    # Derivatives
    def diff_spatialderivative(x, *ops):
        (f, fp), (i, ip) = ops
        ufl_assert(ip is None, "TODO: What happens if ip != 0, i.e. v depends the differentiation variable?")
        # TODO: Are there any issues with indices here? Not sure, think through it...
        if isinstance(fp, spatially_constant_types):
            sh = fp.shape()
            fi = tuple(set(f.free_indices()) ^ set(i))
            ufl_assert(not fi, "FIXME: I think we need free indices in Zero as well...") 
            xp = Zero(sh) 
        else:
            xp = type(x)(fp, i)
        return (x, xp)
    d[SpatialDerivative] = diff_spatialderivative
    
    def diff_variablederivative(x, *ops):
        (f, fp), (v, vp) = ops
        ufl_assert(isinstance(vp, Zero), "TODO: What happens if vp != 0, i.e. v depends the differentiation variable?")
        # TODO: Are there any issues with indices here? Not sure, think through it...
        xp = type(x)(fp, v)
        return (x, xp)
    d[VariableDerivative] = diff_variablederivative
    
    def diff_grad(x, *ops):
        ufl_assert(len(ops) == 1, "Logic breach in diff_commute, len(ops) = %d." % len(ops))
        oprime = ops[0][1]
        ufl_assert(oprime.domain() is not None, "FIXME: How can we handle this?") # Currently calling expand_compounds before AD to avoid this
        return (x, type(x)(oprime))
    d[Grad] = diff_grad
    d[Div]  = diff_commute
    d[Curl] = diff_commute
    d[Rot]  = diff_commute
    
    # Conditionals
    def diff_condition(x, *ops):
        return (x, 0)
    d[EQ] = diff_condition
    d[NE] = diff_condition
    d[LE] = diff_condition
    d[GE] = diff_condition
    d[LT] = diff_condition
    d[GT] = diff_condition
    def diff_conditional(x, *ops):
        c, l, r = ops
        if not isinstance(c[1], Zero):
            ufl_warning("Differentiating a conditional with a condition "\
                "that depends on the differentiation variable."\
                "This is probably not a good idea!")
        if isinstance(l[1], Zero) and isinstance(r[1], Zero):
            return (x, Zero(x.shape()))
        return (x, conditional(c[0], l[1], r[1]))
    d[Conditional] = diff_conditional
    
    return d


def compute_diff(expression, var): # FIXME: Is this correct? Don't understand how I was thinking myself now...
    "Differentiate expression w.r.t Variable var."
    ufl_assert(var is None or var.shape() == (), "VariableDerivative w.r.t. nonscalar variable not implemented.")
    
    handlers = diff_handlers()
    
    def diff_diff(x):
        w = compute_diff(x._expression, x._variable)
        wdiff = compute_diff(w, var)
        return (w, wdiff)
    handlers[VariableDerivative] = diff_diff
    
    _1 = IntValue(1)
    def diff_variable(x):
        if x is var:
            return (x, _1)
        else:
            xdiff = compute_diff(x._expression, var)
            return (x, xdiff)
    handlers[Variable] = diff_variable
    
    # FIXME: Use Variable._diff_cache! 
    
    # Wrap compute_diff result in Variable
    e, ediff = transform(expression, handlers)
    if var is None:
        result = e
    else:
        result = ediff
    
    if not isinstance(result, Variable):
        result = Variable(result)
    return result


def compute_variable_derivatives(form):
    "Apply AD to form, expanding all VariableDerivative w.r.t variables."
    domain = extract_domain(form)
    ufl_assert(domain is not None, "Need to know the spatial dimension to compute derivatives.")
    dim = domain2dim[domain]
    def _compute_diff(expression):
        expression = expand_compounds(expression, dim)
        return compute_diff(expression, None)
    return transform_integrands(form, _compute_diff)


def propagate_spatial_derivatives(form):
    """Partially apply automatic differentiation to form
    by propagating spatial derivatives to terminal objects."""
    ufl_assert(not extract_type(form, SpatialDerivative), "propagate_spatial_derivatives not implemented")
    domain = extract_domain(form)
    ufl_assert(domain is not None, "Need to know the spatial dimension to compute derivatives.")
    dim = domain2dim[domain]
    def _compute_diff(expression):
        expression = expand_compounds(expression, dim)
        # FIXME: Implement!
        return expression
    return transform_integrands(form, _compute_diff)


def compute_form_derivative(form, function, basisfunction):
    "Apply AFD (Automatic Function Differentiation) to Form."
    if isinstance(function, tuple):
        # We got a tuple of functions, handle it as functions
        # over components of a mixed element.
        ufl_assert(all(isinstance(w, Function) for w in function),
            "Expecting a tuple of Functions to differentiate w.r.t.")
        if basisfunction is None:
            elements = [w.element() for w in function]
            element = MixedElement(*elements)
            basisfunction = BasisFunctions(element)
        else:
            ufl_assert()
        functions = zip(function, basisfunction)
    else:
        ufl_assert(isinstance(function, Function),
            "Expecting a Function to differentiate w.r.t.")
        if basisfunction is None:
            basisfunction = BasisFunction(function.element())
        functions = [(function, basisfunction)]
    
    handlers = diff_handlers()
    
    def diff_function(x):
        for (w, wprime) in functions:
            if x == w:
                return (w, wprime)
        return (x, Zero(x.shape()))
    handlers[Function] = diff_function
    
    def diff_variable(x):
        dummy, wprime = transform(x._expression, handlers)
        return (x, wprime)
    handlers[Variable] = diff_variable
    
    domain = form.domain()
    ufl_assert(domain is not None, "Need to know the spatial dimension to compute derivatives.")
    dim = domain2dim[domain]
    
    def _compute_derivative(expression):
        expression = expand_compounds(expression, dim)
        F, J = transform(expression, handlers)
        return J
    
    return transform_integrands(form, _compute_derivative)
