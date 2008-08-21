"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-08-21"

from collections import defaultdict

from ..common import some_key, product
from ..output import ufl_assert, ufl_error

# All classes:
from ..base import UFLObject, Terminal, Number
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunctions import BasisFunction, Function, Constant
#from ..basisfunctions import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index
#from ..indexing import FixedIndex, AxisType, as_index, as_index_tuple, extract_indices
from ..tensors import ListVector, ListMatrix, Tensor
#from ..tensors import Vector, Matrix
from ..algebra import Sum, Product, Division, Power, Mod, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral
#from ..formoperators import Derivative, Action, Rhs, Lhs # TODO: What to do with these?

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients, indices

def transform_integrands(a, transformation):
    """Transform all integrands in a form with a transformation function.
    
    Example usage:
      b = transform_integrands(a, flatten)
    """
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    integrals = []
    for itg in a.integrals():
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
    return handlers[expression.__class__](expression, *ops)


def ufl_reuse_handlers():
    """This function constructs a handler dict for transform
    which can be used to reconstruct a ufl expression through
    transform(...). Nonterminal objects are reused if possible."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in ufl_reuse_handlers. Add to classes.py." % x.__class__)
    d = defaultdict(not_implemented)
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
            return x.__class__(*ops)
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
    def not_implemented(x, ops):
        ufl_error("No handler defined for %s in ufl_copy_handlers. Add to classes.py." % x.__class__)
    d = defaultdict(not_implemented)
    # Terminal objects are simply reused:
    def this(x):
        return x
    for c in terminal_classes:
        d[c] = this
    # Non-terminal objects are reused if all their children are untouched
    def reconstruct(x, *ops):
        return x.__class__(*ops)
    for c in nonterminal_classes:
        d[c] = reconstruct
    return d


def ufl2ufl(expression):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    handlers = ufl_reuse_handlers()
    return transform(expression, handlers)


def ufl2uflcopy(expression):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    handlers = ufl_copy_handlers()
    return transform(expression, handlers)


def latex_handlers():
    # Show a clear error message if we miss some types here:
    def not_implemented(x):
        ufl_error("No handler defined for %s in latex_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    # Utility for parentesizing string:
    def par(s, condition=True):
        if condition:
            return "\\left(%s\\right)" % s
        return str(s)
    # Terminal objects:
    d[Number]        = lambda x: "{%s}" % x._value
    d[BasisFunction] = lambda x: "{v^{%d}}" % x._count # Using ^ for function numbering and _ for indexing
    d[Function]      = lambda x: "{w^{%d}}" % x._count
    d[Constant]      = lambda x: "{w^{%d}}" % x._count
    d[FacetNormal]   = lambda x: "n"
    d[Identity]      = lambda x: "I"
    def l_variable(x, a):
        return "\\left{%s\\right}" % a
    d[Variable]  = l_variable # TODO: Should store expression some place perhaps? LaTeX can express variables!
    def l_multiindex(x):
        return "".join("i_{%d}" % ix._count for ix in x._indices)
    d[MultiIndex] = l_multiindex
    # Non-terminal objects:
    def l_sum(x, *ops):
        return " + ".join(par(o) for o in ops)
    def l_product(x, *ops):
        return " ".join(par(o) for o in ops)
    def l_binop(opstring):
        def particular_l_binop(x, a, b):
            return "{%s}%s{%s}" % (par(a), opstring, par(b))
        return particular_l_binop
    d[Sum]       = l_sum
    d[Product]   = l_product
    d[Division]  = lambda x, a, b: r"\frac{%s}{%s}" % (a, b)
    d[Power]     = l_binop("^")
    d[Mod]       = l_binop("\\mod")
    d[Abs]       = lambda x, a: "|%s|" % a
    d[Transposed] = lambda x, a: "{%s}^T" % a
    d[Indexed]   = lambda x, a, b: "{%s}_{%s}" % (a, b)
    d[PartialDerivative] = lambda x, f, y: "\\frac{\\partial\\left[{%s}\\right]}{\\partial{%s}}" % (f, y)
    #d[Diff] = Diff # FIXME
    d[Grad] = lambda x, f: "\\nabla{%s}" % par(f)
    d[Div]  = lambda x, f: "\\nabla{\\cdot %s}" % par(f)
    d[Curl] = lambda x, f: "\\nabla{\\times %s}" % par(f)
    d[Rot]  = lambda x, f: "\\rot{%s}" % par(f)
    d[MathFunction]  = lambda x, f: "%s%s" % (x._name, par(f)) # FIXME: Add particular functions here
    d[Outer] = l_binop("\\otimes")
    d[Inner] = l_binop(":")
    d[Dot]   = l_binop("\\cdot")
    d[Cross] = l_binop("\\times")
    d[Trace] = lambda x, A: "tr{%s}" % par(A)
    d[Determinant] = lambda x, A: "det{%s}" % par(A)
    d[Inverse]     = lambda x, A: "{%s}^{-1}" % par(A)
    d[Deviatoric]  = lambda x, A: "dev{%s}" % par(A)
    d[Cofactor]    = lambda x, A: "cofac{%s}" % par(A)
    #d[ListVector]  =  FIXME
    #d[ListMatrix]  =  FIXME
    #d[Tensor]      =  FIXME
    d[PositiveRestricted] = lambda x, f: "{%s}^+" % par(A)
    d[NegativeRestricted] = lambda x, f: "{%s}^-" % par(A)
    #d[EQ] = FIXME
    #d[NE] = FIXME
    #d[LE] = FIXME
    #d[GE] = FIXME
    #d[LT] = FIXME
    #d[GT] = FIXME
    #d[Conditional] = FIXME
    
    # Print warnings about classes we haven't implemented:
    missing_handlers = set(ufl_classes)
    missing_handlers.difference_update(d.keys())
    if missing_handlers:
        ufl_warning("In ufl.algorithms.latex_handlers: Missing handlers for classes:\n{\n%s\n}" % \
                    "\n".join(str(c) for c in sorted(missing_handlers)))
    return d


def ufl2latex(expression):
    """Convert an UFL expression to a LaTeX string. Very crude approach."""
    handlers = latex_handlers()
    if isinstance(expression, Form):
        integral_strings = []
        for itg in expression.cell_integrals():
            integral_strings.append(ufl2latex(itg))
        for itg in expression.exterior_facet_integrals():
            integral_strings.append(ufl2latex(itg))
        for itg in expression.interior_facet_integrals():
            integral_strings.append(ufl2latex(itg))
        b = ", ".join("v_{%d}" % i for i,v in enumerate(basisfunctions(expression)))
        c = ", ".join("w_{%d}" % i for i,w in enumerate(coefficients(expression)))
        arguments = "; ".join((b, c))
        latex = "a(" + arguments + ") = " + "  +  ".join(integral_strings)
    elif isinstance(expression, Integral):
        itg = expression
        domain_string = { "cell": "\\Omega",
                          "exterior facet": "\\Gamma^{ext}",
                          "interior facet": "\\Gamma^{int}",
                        }[itg._domain_type]
        integrand_string = transform(itg._integrand, handlers)
        latex = "\\int_{\\Omega_%d} \\left[ %s \\right] \,dx" % (itg._domain_id, integrand_string)
    else:
        latex = transform(expression, handlers)
    return latex


def expand_compounds(expression, dim):
    """Convert an UFL expression to a new UFL expression, with all 
    compound operator objects converted to basic (indexed) expressions."""
    d = ufl_reuse_handlers()
    def e_compound(x, *ops):
        return x.as_basic(dim, *ops)
    for c in compound_classes:
        d[c] = e_compound
    return transform(expression, d)


# TODO: Take care when using this, it will replace _all_ occurences of these indices,
# so in f.ex. (a[i]*b[i]*(1.0 + c[i]*d[i]) the replacement i==0 will result in
# (a[0]*b[0]*(1.0 + c[0]*d[0]) which is probably not what is wanted.
# If this is a problem, a new algorithm may be needed.
def substitute_indices(expression, indices, values):
    """Substitute Index objects from the list 'indices' with corresponding
    fixed values from the list 'values' in expression."""
    d = ufl_reuse_handlers()

    def s_multi_index(x, *ops):
        newindices = []
        for i in x._indices:
            try:
                idx = indices.index(i)
                val = values[idx]
                newindices.append(val)
            except:
                newindices.append(i)
        return MultiIndex(*newindices)
    d[MultiIndex] = s_multi_index

    return transform(expression, d)


def expand_indices(expression):
    "Expand implicit summations into explicit Sums of Products."
    d = ufl_reuse_handlers()
    
    def e_product(x, *ops):
        rep_ind = x._repeated_indices
        return x.__class__(*ops) # FIXME 
    d[Product] = e_product
    
    def e_partial_diff(x, *ops):
        return x # FIXME
    d[PartialDiff] = e_partial_diff
    
    def e_diff(x, *ops):
        return x # FIXME
    d[Diff] = e_diff
    
    return transform(expression, d)


def strip_variables(expression, handled_variables=None):
    d = ufl_reuse_handlers()
    if handled_variables is None:
        handled_variables = {}
    def s_variable(x):
        if x._count in handled_variables:
            return handled_variables[x._count]
        v = strip_variables(x._expression, handled_variables)
        handled_variables[x._count] = v
        return v
    d[Variable] = s_variable
    return transform(expression, d)


def flatten(expression):
    """Convert an UFL expression to a new UFL expression, with sums 
    and products flattened from binary tree nodes to n-ary tree nodes."""
    d = ufl_reuse_handlers()
    def _flatten(x, *ops):
        c = x.__class__
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


def renumber_indices(expression, offset=0):
    "Given an expression, renumber indices in a contiguous count beginning with offset."
    ufl_assert(isinstance(expression, UFLObject), "Expecting an UFLObject.")
    
    # Build a set of all indices used in expression
    idx = indices(expression)
    
    # Build an index renumbering mapping
    k = offset
    indexmap = {}
    for i in idx:
        if i not in indexmap:
            indexmap[i] = Index(count=k)
            k += 1
    
    # Apply index mapping
    handlers = ufl_reuse_handlers()
    def multi_index_handler(o):
        ind = []
        for i in o._indices:
            if isinstance(i, Index):
                ind.append(indexmap[i])
            else:
                ind.append(i)
        return MultiIndex(tuple(ind), len(o._indices))
    handlers[MultiIndex] = multi_index_handler
    return transform(expression, handlers)


def renumber_arguments(a):
    "Given a Form, renumber function and basisfunction count to contiguous sequences beginning at 0."
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    
    # Build sets of all basisfunctions and functions used in expression
    bf = basisfunctions(a)
    cf = functions(a)
    
    # Build a count renumbering mapping for basisfunctions
    bfmap = {}
    k = 0
    for f in bf:
        if f not in bfmap:
            bfmap[f] = BasisFunction(f.element(), count=k)
            k += 1
    
    # Build a count renumbering mapping for coefficients
    cfmap = {}
    k = 0
    for f in cf:
        if f not in cfmap:
            cfmap[f] = Function(element=f._element, name=f._name, count=k)
            k += 1
    
    # Build handler dict using these mappings
    d = ufl_reuse_handlers()
    def basisfunction_handler(o):
        return bfmap[o]
    def function_handler(o):
        return cfmap[o]
    d[BasisFunction] = basisfunction_handler
    d[Function] = function_handler
    
    # Apply renumbering transformation to all integrands 
    def renumber_expression(expression):
        return transform(expression, d)
    return transform_integrands(a, renumber_expression)


def _split_by_dependencies(expression, stacks, variable_cache, terminal_deps):

    if isinstance(expression, Variable):
        c = expression._count
        if c in variable_cache:
            return variable_cache[c]
        #stacks[vdeps].append(v)
    elif isinstance(expression, Terminal):
        deps = terminal_deps[expression.__class__](expression)
        ufl_assert(len(deps) == len(some_key(stacks)), "Inconsistency in dependency definitions.")
        return expression, deps
    
    ops = expression.operands()
    ops2 = [_split_by_dependencies(o, stacks, variable_cache, terminal_deps) for o in ops]
    ops3 = []
    deps = tuple([any(o[1][i] for o in ops2) for i in range(len(ops2[1]))])
    for (v,vdeps) in ops2:
        if isinstance(v, Variable):
            # if this subexpression is a variable, it has already been added to the stack
            ufl_assert(v._count in variable_cache, "")
        elif not vdeps == deps:
            # if this subexpression has other dependencies than 'expression', store a variable for it
            v = Variable(v)
            variable_cache[v._count] = (v, vdeps)
            stacks[vdeps].append(v)
        ops3.append(v)
    
    if isinstance(expression, Variable):
        c = expression._count
        ufl_assert(c not in variable_cache, "Shouldn't reach this point if the variable was already cached!")
        variable_cache[c] = (expression, deps)
        stacks[deps].append(expression)
    
    # Try to reuse expression if nothing has changed:
    if any((o1 is not o3) for (o1,o3) in izip(ops,ops3)):
        e = expression.__class__(*ops3)
    else:
        e = expression
    return e, deps


def split_by_dependencies(expression, basisfunction_deps, function_deps):
    """Split an expression into stacks of variables based on the dependencies of its subexpressions.
    
    @type expression: UFLObject
    @param expression: The expression to parse.
    @type basisfunction_deps: list(tuple(bool,bool))
    @param basisfunction_deps:
        A list of tuples of two booleans, one tuple for each
        BasisFunction in the form the expression originates from.
        Each tuple tells whether this BasisFunction depends on
        the geometry and topology of a cell, respectively.
    @type basisfunction_deps: list(tuple(bool,bool))
    @param function_deps:
        A list of tuples of two booleans, one tuple for each
        Function in the form the expression originates from.
        Each tuple tells whether this Function depends on
        the geometry and topology of a cell, respectively.
    @return: 
        TODO: Describe datastructures
    @precondition:
        Assumes the basisfunctions and functions have been renumbered from 0!
        
    If the *_deps arguments are unknown, a safe way to invoke this function is::
    
        split_by_dependencies(expression, [(False,False)]*rank, [(False,False)]*num_coefficients)
    """
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject.")
    
    ### Dependency tuple definitions
    num_basisfunctions = len(basisfunction_deps)
    num_functions = len(function_deps)
    # Base dependency groups: cell geometry, cell topology, coefficients
    num_base_deps = 3
    # More dependencies: basisfunctions
    num_deps = num_base_deps + num_basisfunctions
    # Utility function to ensure consistent ordering of dependency tuples
    def make_deps(geometry=False, topology=False, coefficients=False, basisfunction=None):
        return (geometry, topology, coefficients) + tuple([False if i == basisfunction else True for i in range(num_basisfunctions)])
    
    ### Stacks: one stack of variables for each dependency configuration
    stacks = {}
    tmp = compute_indices((2,)*num_deps)
    # add empty lists to stacks for each permutation of dependency groups
    permutations = [tuple([bool(i) for i in p]) for p in tmp]
    for p in permutations:
        stacks[p] = []
    
    ### Variable cache, a place to look up if a variable has been added to the stacks already:
    variable_cache = {}
    
    ### Terminal object dependency mappings:
    terminal_deps = {}
    _no_dep = make_deps()
    def no_deps(x):
        return _no_dep
    _facet_normal_dep = make_deps(geometry=True, topology=True)
    def facet_normal_deps(x):
        return _cell_dep
    def function_deps(x):
        g, t = function_deps[x._count]
        return make_deps(geometry=g, topology=t, coefficients=True)
    def basisfunction_deps(x):
        g, t = basisfunction_deps[x._count]
        return make_deps(geometry=g, topology=t, basisfunction=x._count)
    # List all terminal objects:
    terminal_deps[MultiIndex] = no_deps
    terminal_deps[Identity] = no_deps
    terminal_deps[Constant] = no_deps
    terminal_deps[Number] = no_deps
    terminal_deps[FacetNormal] = facet_normal_deps
    terminal_deps[Function] = function_deps
    terminal_deps[BasisFunction] = basisfunction_deps

    ### Do the work!
    e, deps = _split_by_dependencies(expression, stacks, variable_cache)
    # Add final e to stacks and return variable
    if not isinstance(e, Variable):
        e = Variable(e)
    c = e._count
    if c not in variable_cache:
        variable_cache[c] = (e, deps)
        stacks[deps].append(e)
    return e, deps, stacks, variable_cache
