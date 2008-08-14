"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-08-14"

from ..all import * # FIXME
from .analysis import basisfunctions, coefficients

def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    return handlers[expression.__class__](expression, *ops)

def ufl_handlers():
    """This function constructs a handler dict for transform
    which can be used to reconstruct a ufl expression through
    transform(...), which of course makes no sense but is
    useful for testing."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, ops):
        ufl_error("No handler defined for %s in ufl_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    # Terminal objects are simply reused:
    def this(x):
        return x
    d[Number]        = this
    d[BasisFunction] = this
    d[Function]      = this
    d[Constant]      = this
    d[FacetNormal]   = this
    d[MeshSize]      = this
    d[Identity]      = this
    # The classes of non-terminal objects should already have appropriate constructors:
    def construct(x, *ops):
        return x.__class__(*ops)
    d[Variable]  = construct # NB! Some algorithms will not want to move into the expression referenced by a variable.
    d[Sum]       = construct
    d[Product]   = construct
    d[Division]  = construct
    d[Power]     = construct
    d[Mod]       = construct
    d[Abs]       = construct
    d[Transpose] = construct
    d[Indexed]   = construct
    d[PartialDerivative] = construct
    d[Diff] = construct
    d[Grad] = construct
    d[Div]  = construct
    d[Curl] = construct
    d[Rot]  = construct
    d[FiniteElement] = construct
    d[MixedElement]  = construct
    d[VectorElement] = construct
    d[TensorElement] = construct
    d[MathFunction]  = construct
    d[Outer] = construct
    d[Inner] = construct
    d[Dot]   = construct
    d[Cross] = construct
    d[Trace] = construct
    d[Determinant] = construct
    d[Inverse]     = construct
    d[Deviatoric]  = construct
    d[Cofactor]    = construct
    d[ListVector]  = construct
    d[ListMatrix]  = construct
    d[Tensor]      = construct
    return d

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
    d[Number] = lambda x: "{%s}" % str(x._value)
    d[BasisFunction] = lambda x: "{v^{%d}}" % x._count # Using ^ for function numbering and _ for indexing
    d[Function] = lambda x: "{w^{%d}}" % x._count
    d[Constant] = lambda x: "{w^{%d}}" % x._count
    d[FacetNormal] = lambda x: "n"
    d[MeshSize] = lambda x: "h"
    d[Identity] = lambda x: "I"
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
    def l_variable(x, a):
        return "\\left{%s\\right}" % a
    d[Variable]  = l_variable
    d[Sum]       = l_sum
    d[Product]   = l_product
    d[Division]  = lambda x, a, b: r"\frac{%s}{%s}" % (a, b)
    d[Power]     = l_binop("^")
    d[Mod]       = l_binop("\\mod")
    d[Abs]       = lambda x, a: "|%s|" % a
    d[Transpose] = lambda x, a: "{%s}^T" % a
    d[Indexed]   = lambda x, a, b: "{%s}_{%s}" % (a, b)
    d[PartialDerivative] = lambda x, f, y: "\\frac{\\partial\\left[{%s}\\right]}{\\partial{%s}}" % (f, y)
    #d[Diff] = Diff # FIXME
    d[Grad] = lambda x, f: "\\nabla{%s}" % par(f)
    d[Div]  = lambda x, f: "\\nabla{\\cdot %s}" % par(f)
    d[Curl] = lambda x, f: "\\nabla{\\times %s}" % par(f)
    d[Rot]  = lambda x, f: "\\rot{%s}" % par(f)
    #d[FiniteElement] = FiniteElement # Shouldn't be necessary to handle here
    #d[MixedElement]  = MixedElement  # ...
    #d[VectorElement] = VectorElement # ...
    #d[TensorElement] = TensorElement # ...
    d[MathFunction]  = lambda x, f: "%s%s" % (x._name, par(f))
    d[Outer] = l_binop("\\otimes")
    d[Inner] = l_binop(":")
    d[Dot]   = l_binop("\\cdot")
    d[Cross] = l_binop("\\times")
    d[Trace] = lambda x, A: "tr{%s}" % par(A)
    d[Determinant] = lambda x, A: "det{%s}" % par(A)
    d[Inverse]     = lambda x, A: "{%s}^{-1}" % par(A)
    d[Deviatoric]  = lambda x, A: "dev{%s}" % par(A)
    d[Cofactor]    = lambda x, A: "cofac{%s}" % par(A)
    #d[ListVector]  = ListVector # FIXME
    #d[ListMatrix]  = ListMatrix # FIXME
    #d[Tensor]      = Tensor # FIXME
    return d

def ufl2ufl(expression):
    """Convert an UFL expression to a new UFL expression, with no changes.
    Simply tests that objects in the expression behave as expected."""
    handlers = ufl_handlers()
    return transform(expression, handlers)

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

def flatten(expression):
    """Convert an UFL expression to a new UFL expression, with sums 
    and products flattened from binary tree nodes to n-ary tree nodes."""
    handlers = ufl_handlers()
    def _flatten(x, *ops):
        newops = []
        for o in ops:
            if isinstance(x.__class__, o):
                newops.extend(o.operands())
            else:
                newops.append(o)
        return x.__class__(*newops)
    handlers[Sum] = _flatten
    handlers[Product] = _flatten
    return transform(expression, handlers)

def expand_compounds(expression):
    """Convert an UFL expression to a new UFL expression, with 
    compound operator objects converted to indexed expressions."""
    handlers = ufl_handlers()
    def e_outer(x, a, b):
        ii = tuple(Index() for kk in range(a.rank()))
        jj = tuple(Index() for kk in range(b.rank()))
        return a[ii]*b[jj]
    def e_inner(x, a, b):
        ii = tuple(Index() for jj in range(a.rank()))
        return a[ii]*b[ii]
    def e_dot(x, a, b):
        ii = Index()
        if a.rank() == 1: aa = a[ii]
        else: aa = a[...,ii]
        if b.rank() == 1: bb = b[ii]
        else: bb = b[ii,...]
        return aa*bb
    def e_transpose(x, a):
        ii = Index()
        jj = Index()
        return Tensor(a[ii, jj], (jj, ii))
    def e_div(x, a):
        ii = Index()
        if a.rank() == 1: aa = a[ii]
        else: aa = a[...,ii]
        return aa.dx(ii)
    def e_grad(x, a):
        ii = Index()
        if a.rank() > 0:
            jj = tuple(Index() for kk in range(a.rank()))
            return Tensor(a[jj].dx(ii), tuple((ii,)+jj))
        else:
            return Tensor(a.dx(ii), (ii,))
    def e_curl(x, a):
        ufl_error("Not implemented.")
    def e_rot(x, a):
        ufl_error("Not implemented.")
    handlers[Outer] = e_outer
    handlers[Inner] = e_inner
    handlers[Dot]   = e_dot
    handlers[Transpose] = e_transpose
    handlers[Div]  = e_div
    handlers[Grad] = e_grad
    handlers[Curl] = e_curl
    handlers[Rot]  = e_rot
    # FIXME: Some more tensor operations like Det, Dev, etc...,
    # FIXME: diff w.r.t. tensor valued variable?
    return transform(expression, handlers)


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


def _strip_variables(a):
    "Auxilliary procedure for strip_variables."
    
    if isinstance(a, Terminal):
        return a, False
    
    if isinstance(a, Variable):
        b, changed = _strip_variables(a._expression)
        return b, changed
    
    operands = []
    changed = False
    for o in a.operands():
        b, c = _strip_variables(o)
        operands.append(b)
        if c: changed = True
    
    if changed:
        return a.__class__(*operands), True
    # else: no change, reuse object
    return a, False

def strip_variables(a):
    """Strip Variable objects from a, replacing them with their expressions."""
    ufl_assert(isinstance(a, UFLObject), "Expecting an UFLObject.")
    b, changed = _strip_variables(a)
    return b

# naive version, producing lots of extra objects:
def strip_variables2(a):
    """Strip Variable objects from a, replacing them with their expressions."""
    ufl_assert(isinstance(a, UFLObject), "Expecting an UFLObject.")
    
    if isinstance(a, Terminal):
        return a
    
    if isinstance(a, Variable):
        return strip_variables2(a._expression)
    
    operands = [strip_variables2(o) for o in a.operands()]
    
    return a.__class__(*operands)

def flatten(a):
    """Flatten (a+b)+(c+d) into a (a+b+c+d) and (a*b)*(c*d) into (a*b*c*d)."""
    ufl_assert(isinstance(a, UFLObject), "Expecting an UFLObject.")
    
    # Possible optimization:
    #     Reuse objects for subtrees with no
    #     flattened sums or products.
    #     The current procedure will create a new object
    #     for every single node in the tree.
    
    # TODO: Traverse variables or not?
    
    if isinstance(a, Terminal):
        return a
    
    myclass = a.__class__
    operands = []
    
    if isinstance(a, (Sum, Product)):
        for o in a.operands():
            b = flatten(o)
            if isinstance(b, myclass):
                operands.extend(b.operands())
            else:
                operands.append(b)
    else:
        for o in a.operands():
            b = flatten(o)
            operands.append(b)
    
    return myclass(*operands)

def renumber_indices(a):
    "Renumber indices in a contiguous count."
    
    ufl_warning("Not implemented!") # FIXME
    
    # 1) Get all indices
    # 2) Define a index number mapping
    # 3) Apply number map
    
    return a

def renumber_basisfunctions(a):
    "Renumber indices in a contiguous count."
    
    ufl_warning("Not implemented!") # FIXME
    
    # 1) Get all basisfunctions
    # 2) Define a basisfunction number mapping
    # 3) Apply number map
    
    return a

def renumber_functions(a):
    "Renumber indices in a contiguous count."
    
    ufl_warning("Not implemented!") # FIXME
    
    # 1) Get all functions
    # 2) Define a function number mapping
    # 3) Apply number map
    
    return a


def criteria_not_argument(a):
    return not isinstance(a, (Function, BasisFunction))

def criteria_not_trial_function(a):
    return not (isinstance(a, BasisFunction) and (a._count > 0 or a._count == -1))

def criteria_not_basis_function(a):
    return not isinstance(a, BasisFunction)

def _detect_argument_dependencies(a, criteria):
    """Detect edges in expression tree where subtrees
    depend on different stages of form arguments.
    A Variable object is inserted at each edge.
    
    Stage 0:  Subtrees that does not depend on any arguments.
    Stage 1:  Subtrees that does not depend on any basisfunctions (i.e., that only depend on coefficients).
    Stage 2:  Subtrees that does not depend on basisfunction 1 (i.e. trial function for a matrix)
    Stage 3:  Subtrees that does not depend on basisfunction 0 (i.e. test function)
    """
    ufl_warning("NB! Assumes renumbered basisfunctions! FIXME: Implement basisfunction renumbering.")
    
    if isinstance(a, Terminal):
        return a, criteria(a)
    
    operands = []
    crit = []
    for o in a.operands():
        b, c = _detect_argument_dependencies(o, criteria)
        operands.append(b)
        crit.append(c)
    
    # FIXME: finish this
    
    if False:
        return a   
    return a.__class__(*operands)   

def substitute_indices(u, indices, values):
    "Substitute Index objects from list indices with corresponding fixed values from list values in expression u."
    ufl_error("Not implemented") # FIXME: Implement
    return u

# FIXME: Maybe this is better implemented a different way? Could be a good idea to insert Variable instances around expanded sums.
def expand_indices(expression):
    """Convert an UFL expression to a new UFL expression, with 
    compound operator objects converted to indexed expressions."""
    handlers = ufl_handlers()
    def e_product(x, *ops):
        # TODO: Find all repeated indices:
        ii = ()
        a = x
        return Product(*ops) # FIXME
    def e_pdiff(x, *ops):
        return PartialDerivative(*ops) # FIXME
    handlers[Product] = e_product
    handlers[PartialDerivative] = e_pdiff
    # FIXME: Handle all other necessary objects here, f.ex. the Diff object h resulting from "g = variable(g); f = foo(g); h = diff(f[i], g[i])"
    return transform(expression, handlers)

def discover_indices(u):
    "Convert explicit sums into implicit sums (repeated indices)."
    ufl_error("Not implemented") # FIXME: Implement (this is like FFCs 'simplify' done by Marie)
    return u

