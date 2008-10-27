"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-27"

from itertools import izip

from ..output import ufl_assert, ufl_error
from ..common import UFLTypeDefaultDict, UFLTypeDict

# All classes:
from ..base import UFLObject, Terminal, FloatValue, ZeroType
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunction import BasisFunction
from ..function import Function, Constant
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index, FixedIndex, indices, compare_shapes
from ..tensors import ListTensor, ComponentTensor, as_tensor, as_matrix
from ..algebra import Sum, Product, Division, Power, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor, Skew
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import SpatialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes

# Other algorithms:
from .analysis import extract_basisfunctions, extract_coefficients

def transform_integrands(a, transformation):
    """Transform all integrands in a form with a transformation function.
    
    Example usage:
      b = transform_integrands(a, flatten)
    """
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    integrals = []
    for itg in a._integrals:
        integrand = transformation(itg._integrand)
        newitg = Integral(itg._domain_type, itg._domain_id, integrand)
        integrals.append(newitg)
    
    return Form(integrals)

def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    c = expression._uflid
    h = handlers.get(c, None)
    if c is None:
        ufl_error("Didn't find class %s among handlers." % c)
    return h(expression, *ops)

def ufl_reuse_handlers():
    """This function constructs a handler dict for transform
    which can be used to reconstruct a ufl expression through
    transform(...). Nonterminal objects are reused if possible."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in ufl_reuse_handlers. Add to classes.py." % x._uflid)
    d = UFLTypeDefaultDict(not_implemented)
    
    # Terminal objects are simply reused:
    def this(x):
        return x
    for c in terminal_classes:
        d[c] = this
    # Non-terminal objects are reused if all their children are untouched
    def reconstruct(x, *ops):
        if all((a is b) for (a,b) in izip(x.operands(), ops)):
            return x
        else:
            return type(x)(*ops)
    for c in nonterminal_classes:
        d[c] = reconstruct
    return d

def ufl_copy_handlers():
    """This function constructs a handler dict for transform
    which can be used to reconstruct a ufl expression through
    transform(...). Nonterminal objects are copied, such that 
    no nonterminal objects are shared between the new and old
    expression."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in ufl_copy_handlers. Add to classes.py." % x._uflid)
    d = UFLTypeDefaultDict(not_implemented)

    # Terminal objects are simply reused:
    def this(x):
        return x
    for c in terminal_classes:
        d[c] = this
    # Non-terminal objects are reused if all their children are untouched
    def reconstruct(x, *ops):
        return type(x)(*ops)
    for c in nonterminal_classes:
        d[c] = reconstruct
    return d

def ufl2ufl(expression):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject.")
    handlers = ufl_reuse_handlers()
    return transform(expression, handlers)

def ufl2uflcopy(expression):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject.")
    handlers = ufl_copy_handlers()
    return transform(expression, handlers)

def flatten(expression):
    """Convert an UFL expression to a new UFL expression, with sums 
    and products flattened from binary tree nodes to n-ary tree nodes."""
    d = ufl_reuse_handlers()
    def _flatten(x, *ops):
        c = x._uflid
        newops = []
        for o in ops:
            if isinstance(o, c):
                newops.extend(o.operands())
            else:
                newops.append(o)
        return c(*newops)
    d[Sum] = _flatten
    d[Product] = _flatten
    return transform(expression, d)

def replace(expression, substitution_map):
    """Replace objects in expression.
    
    @param expression:
        A UFLObject.
    @param substitution_map:
        A dict with from:to replacements to perform.
    """
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject.")
    handlers = ufl_reuse_handlers()
    orig_handlers = UFLTypeDict()
    
    def r_replace(x, *ops):
        y = substitution_map.get(x)
        if y is None:
            c = x._uflid
            h = orig_handlers[c]
            return h(x, *ops)
        return y
    
    for k in substitution_map.keys():
        c = k._uflid
        orig_handlers[c] = handlers[c]
        handlers[c] = r_replace
    
    return transform(expression, handlers)

def replace_in_form(form, substitution_map):
    "Apply replace to all integrands in a form."
    ufl_assert(isinstance(form, Form), "Expecting Form.")
    def replace_expression(expression):
        return replace(expression, substitution_map)
    return transform_integrands(form, replace_expression)

def expand_compounds_handlers(dim, variables):
    # Note:
    # To avoid typing errors, the expressions for cofactor and deviatoric parts 
    # below were created with the script tensoralgebrastrings.py under ufl/scripts/
    
    d = ufl_reuse_handlers()
    
    def e_grad(x, f):
        ii = Index()
        if f.rank() > 0:
            jj = tuple(indices(f.rank()))
            return as_tensor(f[jj].dx(ii), tuple((ii,)+jj))
        else:
            return as_tensor(f.dx(ii), (ii,))
    d[Grad] = e_grad
    
    def e_div(x, f):
        ii = Index()
        if f.rank() == 1:
            g = f[ii]
        else:
            g = f[...,ii]
        return g.dx(ii)
    d[Div] = e_div
    
    def e_curl(x, f):
        FIXME
    d[Curl] = e_curl
    
    def e_rot(x, f):
        FIXME
    d[Rot] = e_rot
    
    def e_transposed(x, A):
        ii, jj = indices(2)
        return as_tensor(A[ii, jj], (jj, ii))
    d[Transposed] = e_transposed
    
    def e_outer(x, a, b):
        ii = tuple(indices(a.rank()))
        jj = tuple(indices(b.rank()))
        return a[ii]*b[jj]
    d[Outer] = e_outer
    
    def e_inner(x, a, b):
        ii = tuple(indices(a.rank()))
        return a[ii]*b[ii]
    d[Inner] = e_inner
    
    def e_dot(x, a, b):
        ii = Index()
        aa = a[ii] if (a.rank() == 1) else a[...,ii]
        bb = b[ii] if (b.rank() == 1) else b[ii,...]
        return aa*bb
    d[Dot] = e_dot
    
    def e_cross(x, a, b):
        ufl_assert(compare_shapes(a.shape(), (3,)),
            "Invalid shape of first argument in cross product.")
        ufl_assert(compare_shapes(b.shape(), (3,)),
            "Invalid shape of second argument in cross product.")
        def c(i, j):
            return a[i]*b[j]-a[j]*b[i]
        return as_vector((c(1,2), c(2,0), c(0,1)))
    d[Cross] = e_cross
    
    def e_trace(x, A):
        i = Index()
        return A[i,i]
    d[Trace] = e_trace
    
    def e_determinant(x, A):
        sh = complete_shape(A.shape(), dim)

        if len(sh) == 0:
            return A
        
        def det2D(B, i, j, k, l):
            return B[i,k]*B[j,l]-B[i,l]*B[j,k]

        if sh[0] == 2:
            return det2D(A, 0, 1, 0, 1)
        
        if sh[0] == 3:
            # TODO: Verify this expression
            return A[0,0]*det2D(A, 1, 2, 1, 2) + \
                   A[0,1]*det2D(A, 1, 2, 2, 0) + \
                   A[0,2]*det2D(A, 1, 2, 0, 1)
        
        # TODO: Implement generally for all dimensions?
        ufl_error("Determinant not implemented for dimension %d." % dim)
    d[Determinant] = e_determinant
    
    def e_cofactor(x, A):
        sh = complete_shape(A.shape(), dim)
        if sh[0] == 2:
            return as_matrix([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([ \
                [A[2,2]*A[1,1]-A[1,2]*A[2,1],-A[0,1]*A[2,2]+A[0,2]*A[2,1],A[0,1]*A[1,2]-A[0,2]*A[1,1]],
                [-A[2,2]*A[1,0]+A[1,2]*A[2,0],-A[0,2]*A[2,0]+A[2,2]*A[0,0],A[0,2]*A[1,0]-A[1,2]*A[0,0]],
                [A[1,0]*A[2,1]-A[2,0]*A[1,1],A[0,1]*A[2,0]-A[0,0]*A[2,1],A[0,0]*A[1,1]-A[0,1]*A[1,0]] \
                ])
        elif sh[0] == 4:
            return as_matrix([ \
                [-A[3,3]*A[2,1]*A[1,2]+A[1,2]*A[3,1]*A[2,3]+A[1,1]*A[3,3]*A[2,2]-A[3,1]*A[2,2]*A[1,3]+A[2,1]*A[1,3]*A[3,2]-A[1,1]*A[3,2]*A[2,3],-A[3,1]*A[0,2]*A[2,3]+A[0,1]*A[3,2]*A[2,3]-A[0,3]*A[2,1]*A[3,2]+A[3,3]*A[2,1]*A[0,2]-A[3,3]*A[0,1]*A[2,2]+A[0,3]*A[3,1]*A[2,2],A[3,1]*A[1,3]*A[0,2]+A[1,1]*A[0,3]*A[3,2]-A[0,3]*A[1,2]*A[3,1]-A[0,1]*A[1,3]*A[3,2]+A[3,3]*A[1,2]*A[0,1]-A[1,1]*A[3,3]*A[0,2],A[1,1]*A[0,2]*A[2,3]-A[2,1]*A[1,3]*A[0,2]+A[0,3]*A[2,1]*A[1,2]-A[1,2]*A[0,1]*A[2,3]-A[1,1]*A[0,3]*A[2,2]+A[0,1]*A[2,2]*A[1,3]],
                [A[3,3]*A[1,2]*A[2,0]-A[3,0]*A[1,2]*A[2,3]+A[1,0]*A[3,2]*A[2,3]-A[3,3]*A[1,0]*A[2,2]-A[1,3]*A[3,2]*A[2,0]+A[3,0]*A[2,2]*A[1,3],A[0,3]*A[3,2]*A[2,0]-A[0,3]*A[3,0]*A[2,2]+A[3,3]*A[0,0]*A[2,2]+A[3,0]*A[0,2]*A[2,3]-A[0,0]*A[3,2]*A[2,3]-A[3,3]*A[0,2]*A[2,0],-A[3,3]*A[0,0]*A[1,2]+A[0,0]*A[1,3]*A[3,2]-A[3,0]*A[1,3]*A[0,2]+A[3,3]*A[1,0]*A[0,2]+A[0,3]*A[3,0]*A[1,2]-A[0,3]*A[1,0]*A[3,2],A[0,3]*A[1,0]*A[2,2]+A[1,3]*A[0,2]*A[2,0]-A[0,0]*A[2,2]*A[1,3]-A[0,3]*A[1,2]*A[2,0]+A[0,0]*A[1,2]*A[2,3]-A[1,0]*A[0,2]*A[2,3]],
                [A[3,1]*A[1,3]*A[2,0]+A[3,3]*A[2,1]*A[1,0]+A[1,1]*A[3,0]*A[2,3]-A[1,0]*A[3,1]*A[2,3]-A[3,0]*A[2,1]*A[1,3]-A[1,1]*A[3,3]*A[2,0],A[3,3]*A[0,1]*A[2,0]-A[3,3]*A[0,0]*A[2,1]-A[0,3]*A[3,1]*A[2,0]-A[3,0]*A[0,1]*A[2,3]+A[0,0]*A[3,1]*A[2,3]+A[0,3]*A[3,0]*A[2,1],-A[0,0]*A[3,1]*A[1,3]+A[0,3]*A[1,0]*A[3,1]-A[3,3]*A[1,0]*A[0,1]+A[1,1]*A[3,3]*A[0,0]-A[1,1]*A[0,3]*A[3,0]+A[3,0]*A[0,1]*A[1,3],A[0,0]*A[2,1]*A[1,3]+A[1,0]*A[0,1]*A[2,3]-A[0,3]*A[2,1]*A[1,0]+A[1,1]*A[0,3]*A[2,0]-A[1,1]*A[0,0]*A[2,3]-A[0,1]*A[1,3]*A[2,0]],
                [-A[1,2]*A[3,1]*A[2,0]-A[2,1]*A[1,0]*A[3,2]+A[3,0]*A[2,1]*A[1,2]-A[1,1]*A[3,0]*A[2,2]+A[1,0]*A[3,1]*A[2,2]+A[1,1]*A[3,2]*A[2,0],-A[3,0]*A[2,1]*A[0,2]-A[0,1]*A[3,2]*A[2,0]+A[3,1]*A[0,2]*A[2,0]-A[0,0]*A[3,1]*A[2,2]+A[3,0]*A[0,1]*A[2,2]+A[0,0]*A[2,1]*A[3,2],A[0,0]*A[1,2]*A[3,1]-A[1,0]*A[3,1]*A[0,2]+A[1,1]*A[3,0]*A[0,2]+A[1,0]*A[0,1]*A[3,2]-A[3,0]*A[1,2]*A[0,1]-A[1,1]*A[0,0]*A[3,2],-A[1,1]*A[0,2]*A[2,0]+A[2,1]*A[1,0]*A[0,2]+A[1,2]*A[0,1]*A[2,0]+A[1,1]*A[0,0]*A[2,2]-A[1,0]*A[0,1]*A[2,2]-A[0,0]*A[2,1]*A[1,2]] \
                ])
        ufl_error("Cofactor not implemented for dimension %s." % sh[0])
    d[Cofactor] = e_cofactor
    
    def e_inverse(x, A):
        if A.rank() == 0:
            return 1.0 / A
        return e_determinant(A) * e_cofactor(A)
    d[Inverse] = e_inverse
    
    def e_deviatoric(x, A):
        sh = complete_shape(A.shape(), dim)
        if sh[0] == 2:
            return as_matrix([[-A[1,1],A[0,1]],[A[1,0],-A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([[-A[1,1]-A[2,2],A[0,1],A[0,2]],[A[1,0],-A[0,0]-A[2,2],A[1,2]],[A[2,0],A[2,1],-A[0,0]-A[1,1]]])
        ufl_error("dev(A) not implemented for dimension %s." % sh[0])
    d[Deviatoric] = e_deviatoric
    
    def e_skew(x, A):
        i, j = indices(2)
        return as_matrix( (A[i,j] - A[j,i]) / 2, (i,j) )
    d[Skew] = e_skew
    
    def e_variable(x):
        return variables[x._count]
    d[Variable] = e_variable

    return d

def expand_compounds(expression, dim):
    """Convert an UFL expression to a new UFL expression, with all 
    compound operator objects converted to basic (indexed) expressions."""
    
    variables = {}
    handlers = expand_compounds_handlers(dim, variables)
    
    vars = extract_variables(expression)
    for v in vars:
        ufl_assert(v._count not in variables, "Expecting a unique sequence of variables from extract_variables.")
        e = transform(v._expression, handlers)
        v2 = Variable(e, v._count)
        variables[v._count] = v2
    
    return transform(expression, handlers)
