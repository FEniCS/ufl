"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-09-24"

from collections import defaultdict

from ..output import ufl_error, ufl_warning

# All classes:
from ..base import FloatValue
from ..variable import Variable
from ..basisfunction import BasisFunction
from ..function import Function, Constant
#from ..basisfunction import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index, FixedIndex
#from ..indexing import AxisType, as_index, as_index_tuple, extract_indices
from ..tensors import ListVector, ListMatrix, Tensor
#from ..tensors import Vector, Matrix
from ..algebra import Sum, Product, Division, Power, Mod, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import SpatialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients, indices

# General dict-based transformation utility
from .transformations import transform


def latex_handlers():
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in latex_handlers." % x.__class__)
    def make_not_implemented():
        return not_implemented
    d = defaultdict(make_not_implemented)
    # Utility for parentesizing string:
    def par(s, condition=True):
        if condition:
            return "\\left(%s\\right)" % s
        return str(s)
    # Terminal objects:
    d[FloatValue]        = lambda x: "{%s}" % x._value
    d[BasisFunction] = lambda x: "{v^{%d}}" % x._count # Using ^ for function numbering and _ for indexing
    d[Function]      = lambda x: "{w^{%d}}" % x._count
    d[Constant]      = lambda x: "{w^{%d}}" % x._count
    d[FacetNormal]   = lambda x: "n"
    d[Identity]      = lambda x: "I"
    def l_variable(x):
        # TODO: Should store expression some place perhaps? LaTeX can express variables! We can build a sequence of variable assignments s_i = expression if we want.
        return "\\left{%s\\right}" % x._expression
    d[Variable]  = l_variable
    def l_multiindex(x):
        s = ""
        for ix in x._indices:
            if isinstance(ix, FixedIndex):
                s += "%d" % ix._value
            elif isinstance(ix, Index):
                s += "i_{%d}" % ix._count
            else:
                ufl_error("Unknown index type %s." % type(ix))
        return s
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
    d[SpatialDerivative] = lambda x, f, y: "\\frac{\\partial\\left[{%s}\\right]}{\\partial{%s}}" % (f, y)
    def l_diff(x, f, v):
        return r"\frac{d\left[%s\right]}{d\left[%s\right]}" % (f, v)
    d[Diff] = l_diff
    d[Grad] = lambda x, f: "\\nabla{%s}" % par(f)
    d[Div]  = lambda x, f: "\\nabla{\\cdot %s}" % par(f)
    d[Curl] = lambda x, f: "\\nabla{\\times %s}" % par(f)
    d[Rot]  = lambda x, f: "\\rot{%s}" % par(f)
    
    d[Sqrt] = lambda x, f: "\\sqrt{%s}" % par(f)
    d[Exp]  = lambda x, f: "e^{%s}" % par(f)
    d[Ln]   = lambda x, f: "\\ln{%s}" % par(f)
    d[Cos]  = lambda x, f: "\\cos{%s}" % par(f)
    d[Sin]  = lambda x, f: "\\sin{%s}" % par(f)
    
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
        latex = "a(" + arguments + ") = \n    " + "\n    + ".join(integral_strings)
    
    elif isinstance(expression, Integral):
        itg = expression
        domain_string = { "cell": "\\Omega",
                          "exterior_facet": "\\Gamma^{ext}",
                          "interior_facet": "\\Gamma^{int}",
                        }[itg._domain_type]
        dx_string = { "cell": "dx",
                      "exterior_facet": "ds",
                      "interior_facet": "dS",
                    }[itg._domain_type]
        integrand_string = transform(itg._integrand, handlers)
        latex = "\\int_{%s_%d} \\left[ { %s } \\right] \,%s" % (domain_string, itg._domain_id, integrand_string, dx_string)
    else:
        latex = transform(expression, handlers)
    
    return latex


