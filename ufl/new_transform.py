"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-05-16"

from all import *
from utilities import basisfunctions, coefficients

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
    d[Symbol]        = this
    d[BasisFunction] = this
    d[Function]      = this
    d[Constant]      = this
    d[FacetNormal]   = this
    d[MeshSize]      = this
    d[Identity]      = this
    # The classes of non-terminal objects should already have appropriate constructors:
    def construct(x, *ops):
        return x.__class__(*ops)
    d[Variable]  = construct
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
    # Terminal objects:
    d[Number] = lambda x: "{%s}" % str(x._value)
    d[Symbol] = lambda x: "{%s}" % x._name
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
        return " + ".join("(%s)" % o for o in ops)
    def l_product(x, *ops):
        return " * ".join("(%s)" % o for o in ops)
    def l_binop(opstring):
        def particular_l_binop(x, a, b):
            return ("{%s}" % a) + opstring + ("{%s}" % b)
        return particular_l_binop
    #d[Variable]  = Variable # TODO
    d[Sum]       = l_sum
    d[Product]   = l_product
    d[Division]  = lambda x, a, b: r"\frac{%s}{%s}" % (a, b)
    d[Power]     = l_binop("^") #lambda x, a, b: r"{%s}^{%s}" % (a, b)
    d[Mod]       = l_binop("\\mod") #lambda x, a, b: r"{%s}\mod{%s}" % (a, b)
    d[Abs]       = lambda x, a: "|%s|" % a
    d[Transpose] = lambda x, a: "{%s}^T" % a
    d[Indexed]   = lambda x, a, b: "{%s}_{%s}" % (a, b) # TODO: Index handler is missing!
    d[PartialDerivative] = lambda x, f, y: "\\frac{\\partial\\left[{%s}\\right]}{\\partial{%s}}" % (f, y)
    #d[Diff] = Diff # TODO: May be removed?
    d[Grad] = lambda x, f: "\\Nabla (%s)" % f 
    d[Div]  = lambda x, f: "\\Nabla \\cdot (%s)" % f
    d[Curl] = lambda x, f: "\\Nabla \\times (%s)" % f
    d[Rot]  = lambda x, f: "\\rot (%s)" % f
    #d[FiniteElement] = FiniteElement # Shouldn't be necessary to handle here
    #d[MixedElement]  = MixedElement  # ...
    #d[VectorElement] = VectorElement # ...
    #d[TensorElement] = TensorElement # ...
    d[MathFunction]  = lambda x, f: "%s(%s)" % (x._name, f)
    d[Outer] = lambda x, a, b: "(%s) \\otimes (%s)" % (a, b)
    d[Inner] = lambda x, a, b: "(%s) : (%s)" % (a, b)
    d[Dot]   = lambda x, a, b: "(%s) \\cdot (%s)" % (a, b)
    d[Cross] = lambda x, a, b: "(%s) \\times (%s)" % (a, b)
    d[Trace] = lambda x, A: "tr(%s)" % A
    d[Determinant] = lambda x, A: "det(%s)" % A
    d[Inverse]     = lambda x, A: "(%s)^{-1}" % A
    d[Deviatoric]  = lambda x, A: "dev(%s)" % A
    d[Cofactor]    = lambda x, A: "cofac(%s)" % A
    #d[ListVector]  = ListVector # TODO
    #d[ListMatrix]  = ListMatrix # TODO
    #d[Tensor]      = Tensor # TODO
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
    return transform(expression, handlers)


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
    # FIXME: Handle all other necessary objects here
    return transform(expression, handlers)



# These can be implemented in form compilers:

def sfc_handlers():
    # Show a clear error message if we miss some types here:
    def not_implemented(x):
        ufl_error("No handler defined for %s in sfl_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    # FIXME
    return d

def ufl2sfc(expression):
    handlers = sfc_handlers()
    return transform(expression, handlers)


def ffc_handlers():
    # Show a clear error message if we miss some types here:
    def not_implemented(x):
        ufl_error("No handler defined for %s in ffl_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    # FIXME
    return d

def ufl2ffc(expression):
    handlers = ffc_handlers()
    return transform(expression, handlers)

