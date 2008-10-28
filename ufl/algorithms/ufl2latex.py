"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-28"

# Modified by Anders Logg, 2008.

from itertools import chain
from collections import defaultdict

from ..output import ufl_error, ufl_debug, ufl_warning
from ..common import UFLTypeDefaultDict

# All classes:
from ..base import FloatValue, ZeroType
from ..variable import Variable
from ..basisfunction import BasisFunction
from ..function import Function, Constant
#from ..basisfunction import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index, FixedIndex, AxisType
#from ..indexing import as_index, as_index_tuple, extract_indices
from ..tensors import ListTensor, ComponentTensor
from ..algebra import Sum, Product, Division, Power, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import SpatialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes

# Other algorithms:
from .analysis import extract_basisfunctions, extract_coefficients, extract_variables
from .formdata import FormData

# General dict-based transformation utility
from .transformations import transform

def latex_handlers(basisfunction_renumbering, coefficient_renumbering):
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in latex_handlers." % type(x))
    d = UFLTypeDefaultDict(not_implemented)
    
    # Utility for parentesizing string (TODO: Finish precedence handling):
    def par(s, condition=True):
        if condition:
            return "\\left(%s\\right)" % s
        return str(s)
    
    # Terminal objects:
    def l_zero(x):
        if x.shape() == ():
            return "0"
        return r"{\mathbf 0}"
    d[ZeroType]      = l_zero
    d[FloatValue]    = lambda x: "{%s}" % x._value
    d[FacetNormal]   = lambda x: "n"
    d[Identity]      = lambda x: "I"
    d[BasisFunction] = lambda x: "{v^{%d}}" % basisfunction_renumbering[x] # Using ^ for function numbering and _ for indexing
    d[Function]      = lambda x: "{w^{%d}}" % coefficient_renumbering[x]
    d[Constant]      = lambda x: "{w^{%d}}" % coefficient_renumbering[x]
    d[Variable]      = lambda x: "s_{%d}" % x._count
    
    def l_index(ix):
        if isinstance(ix, FixedIndex):
            s = "%d" % ix._value
        elif isinstance(ix, Index):
            s = "i_{%d}" % ix._count
        elif isinstance(ix, AxisType):
            s = ":" # TODO: How to express this in natural syntax?
        else:
            ufl_error("Unknown index type %s." % type(ix))
        return s
    #d[Index] = l_index # not UFLOjbect, used below as helper
    
    def l_multiindex(x):
        return "".join(l_index(ix) for ix in x)
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
    d[Abs]       = lambda x, a: "|%s|" % a
    d[Transposed] = lambda x, a: "{%s}^T" % a
    d[Indexed]   = lambda x, a, b: "{%s}_{%s}" % (a, b)
    
    def l_spatial_diff(x, f, ii):
        ii = x.operands()[1]
        n = len(ii)
        y = "".join("\partial{}x_{%s}" % l_index(i) for i in ii)
        l = r"\left["
        r = r"\right]"
        d = "" if (n == 1) else (r"^%d" % n)
        nom = r"\partial%s%s%s%s" % (l, f, r, d)
        denom = r"%s" % y
        return r"\frac{%s}{%s}" % (nom, denom)
    d[SpatialDerivative] = l_spatial_diff
    
    def l_diff(x, f, v):
        nom = r"\partial\left[%s\right]" % f
        denom = r"\partial\left[%s\right]" % v
        return r"\frac{%s}{%s}" % (nom, denom)
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
    def l_listtensor(*ops):
        return "\\left{FIXME: \\LaTeX handler for ListTensor not implemented!\\right}"
        #return "\\matrix[%s]{%s}" % ("c"*len(ops), "".join("{%s}" % o for o in ops))
    d[ListTensor]  = l_listtensor
    def l_componenttensor(*ops):
        return "\\left{FIXME: \\LaTeX handler for ComponentTensor not implemented!\\right}"
    d[ComponentTensor] = l_componenttensor
    d[PositiveRestricted] = lambda x, f: "{%s}^+" % par(A)
    d[NegativeRestricted] = lambda x, f: "{%s}^-" % par(A)
    d[EQ] = lambda a, b: "(%s = %s)" % (a, b)
    d[NE] = lambda a, b: "(%s \\ne %s)" % (a, b)
    d[LE] = lambda a, b: "(%s \\le %s)" % (a, b)
    d[GE] = lambda a, b: "(%s \\ge %s)" % (a, b)
    d[LT] = lambda a, b: "(%s < %s)" % (a, b)
    d[GT] = lambda a, b: "(%s > %s)" % (a, b)
    d[Conditional] = lambda c, t, f: "\\left{ %s if %s otherwise %s \\right}" % (t, c, f) # FIXME
    
    # Print warnings about classes we haven't implemented:
    missing_handlers = set(ufl_classes)
    missing_handlers.difference_update(d.keys())
    if missing_handlers:
        ufl_debug("In ufl.algorithms.latex_handlers: Missing handlers for classes:\n{\n%s\n}" % \
                    "\n".join(str(c) for c in sorted(missing_handlers)))
    return d

def element2latex(element):
    return "{\mbox{%s}}" % str(element)

def expression2latex(expression, basisfunction_renumbering, coefficient_renumbering):
    handlers = latex_handlers(basisfunction_renumbering, coefficient_renumbering)
    latex = transform(expression, handlers)
    return latex

def form2latex(form, formname="a", newline = " \\\\ \n"):
    formdata = FormData(form)

    ba = "\n\\begin{align}\n"
    ea = "\n\\end{align}\n"

    def make_align(lines):
        if strings:
            latex = ba
            latex += newline.join(lines)
            latex += ea
            return latex
        return ""
    
    latex = ""

    # Define elements
    strings = []
    for i, f in enumerate(formdata.basisfunctions):
        e = f.element()
        strings.append("\\mathcal{P}_{%d} = \{%s\} " % (i, element2latex(e)))
    for i, f in enumerate(formdata.coefficients):
        e = f.element()
        strings.append("\\mathcal{Q}_{%d} = \{%s\} " % (i, element2latex(e)))
    latex += make_align(strings)

    # Define function spaces
    strings = []
    for i, f in enumerate(formdata.basisfunctions):
        strings.append("V_h^{%d} = \{v : v \\vert_K \in \\mathcal{P}_{%d}(K) \\quad \\forall K \in \\mathcal{T}\} " % (i, i))
    for i, f in enumerate(formdata.coefficients):
        strings.append("W_h^{%d} = \{v : v \\vert_K \in \\mathcal{Q}_{%d}(K) \\quad \\forall K \in \\mathcal{T}\} " % (i, i))
    latex += make_align(strings)
    
    # Define basis functions and functions
    # TODO: Get names of arguments from form file
    strings = []
    for i,e in enumerate(formdata.basisfunctions):
        strings.append("v_h^{%d} \\in V_h^{%d} " % (i, i))
    for i,f in enumerate(formdata.coefficients):
        strings.append("w_h^{%d} \\in W_h^{%d} " % (i, i))
    latex += make_align(strings)
    
    # Define variables
    handled_variables = set()
    integrals = list(chain(form.cell_integrals(),
                           form.exterior_facet_integrals(),
                           form.interior_facet_integrals()))
    strings = []
    for itg in integrals:
        vars = extract_variables(itg.integrand())
        for v in vars:
            if not v._count in handled_variables:
                handled_variables.add(v._count)
                exprlatex = expression2latex(v._expression, formdata.basisfunction_renumbering, formdata.coefficient_renumbering)
                strings.append("s_{%d} &= %s " % (v._count, exprlatex))
    latex += make_align(strings)
    
    # Join form arguments for "a(...) ="
    b = ", ".join("v_h^{%d}" % i for (i,v) in enumerate(formdata.basisfunctions))
    c = ", ".join("w_h^{%d}" % i for (i,w) in enumerate(formdata.coefficients))
    arguments = "; ".join((b, c))
    latex += ba
    latex += "%s(%s) = " % (formname, arguments, )

    # Define integrals
    domain_strings = { "cell": "\\Omega",
                       "exterior_facet": "\\Gamma^{ext}",
                       "interior_facet": "\\Gamma^{int}",
                     }
    dx_strings = { "cell": "dx",
                   "exterior_facet": "ds",
                   "interior_facet": "dS",
                 }
    integral_strings = []
    for itg in integrals:
        integrand_string = expression2latex(itg.integrand(), formdata.basisfunction_renumbering, formdata.coefficient_renumbering)
        itglatex = "\\int_{%s_%d} & \\left[ { %s } \\right] \,%s" % \
                (domain_strings[itg._domain_type],
                 itg._domain_id,
                 integrand_string,
                 dx_strings[itg._domain_type])
        integral_strings.append(itglatex)

    # Join integral strings, and we're done!
    latex += (newline + " & {}+ ").join(integral_strings)
    latex += ea
    return latex

def ufl2latex(expression):
    if isinstance(expression, Form):
        return form2latex(expression)
    basisfunction_renumbering = dict((f,f._count) for f in extract_basisfunctions(expression))
    coefficient_renumbering = dict((f,f._count) for f in extract_coefficients(expression))
    return expression2latex(expression, basisfunction_renumbering, coefficient_renumbering)
