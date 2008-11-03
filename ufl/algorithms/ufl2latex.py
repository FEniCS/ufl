"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-11-03"

# Modified by Anders Logg, 2008.

import os
from itertools import chain
from collections import defaultdict

from ..output import ufl_error, ufl_debug, ufl_warning
from ..common import UFLTypeDefaultDict

# All classes:
from ..zero import Zero
from ..scalar import FloatValue, IntValue
from ..variable import Variable
from ..basisfunction import BasisFunction
from ..function import Function, Constant
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index, FixedIndex, AxisType
from ..tensors import ListTensor, ComponentTensor
from ..algebra import Sum, Product, Division, Power, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral

# Lists of all Expr classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes

# Other algorithms:
from .transformations import transform
from .analysis import extract_basisfunctions, extract_coefficients, extract_variables
from .formdata import FormData
from .checks import validate_form
from .formfiles import load_forms

# TODO: Must rewrite LaTeX expression compiler to handle parent before child, to handle line wrapping, ListTensors of rank > 1, +++

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
    d[Zero]      = l_zero
    d[FloatValue]    = lambda x: "{%s}" % x._value
    d[IntValue]      = lambda x: "{%s}" % x._value
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
    d[Sum]       = l_sum

    def l_product(x, *ops):
        return " ".join(par(o) for o in ops)
    d[Product]   = l_product
    
    d[Division]  = lambda x, a, b: r"\frac{%s}{%s}" % (a, b)
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
    d[VariableDerivative] = l_diff
    
    d[Grad] = lambda x, f: "\\nabla{%s}" % par(f)
    d[Div]  = lambda x, f: "\\nabla{\\cdot %s}" % par(f)
    d[Curl] = lambda x, f: "\\nabla{\\times %s}" % par(f)
    d[Rot]  = lambda x, f: "\\rot{%s}" % par(f)
    
    d[Sqrt] = lambda x, f: "%s^{\frac 1 2}" % par(f)
    d[Exp]  = lambda x, f: "e^{%s}" % par(f)
    d[Ln]   = lambda x, f: "\\ln{%s}" % par(f)
    d[Cos]  = lambda x, f: "\\cos{%s}" % par(f)
    d[Sin]  = lambda x, f: "\\sin{%s}" % par(f)
    
    def l_binop(opstring):
        def particular_l_binop(x, a, b):
            return "{%s}%s{%s}" % (par(a), opstring, par(b))
        return particular_l_binop
    d[Power] = l_binop("^")
    d[Outer] = l_binop("\\otimes")
    d[Inner] = l_binop(":")
    d[Dot]   = l_binop("\\cdot")
    d[Cross] = l_binop("\\times")
    
    d[Trace]       = lambda x, A: "tr{%s}" % par(A)
    d[Determinant] = lambda x, A: "det{%s}" % par(A)
    d[Inverse]     = lambda x, A: "{%s}^{-1}" % par(A)
    d[Deviatoric]  = lambda x, A: "dev{%s}" % par(A)
    d[Cofactor]    = lambda x, A: "cofac{%s}" % par(A)
    
    def l_listtensor(x, *ops):
        shape = x.shape()
        if len(shape) == 1:
            l = " \\\\ \n ".join(ops)
        elif len(shape) == 2:
            ufl_error("LaTeX handler for list matrix currently gives transposed output... Need to know parent!")
            l = " & \n ".join(ops)            
        else:
            ufl_error("TODO: LaTeX handler for list tensor of rank 3 or higher not implemented!")
        return "\\left[\\begin{matrix}{%s}\\end{matrix}\\right]^T" % l
    d[ListTensor]  = l_listtensor
    
    def l_componenttensor(x, *ops):
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
    
    def l_conditional(x, c, t, f):
        l = "\\begin{cases}\n"
        l += "%s, &\text{if }\quad %s, \\\\\n" % (t, c)
        l += "%s, &\text{otherwise.}\n" % f
        l += "\\end{cases}"
        return l
    d[Conditional] = l_conditional
    
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
    if strings:
        latex += "Finite elements:\n"
        latex += make_align(strings)

    # Define function spaces
    strings = []
    for i, f in enumerate(formdata.basisfunctions):
        strings.append("V_h^{%d} = \{v : v \\vert_K \in \\mathcal{P}_{%d}(K) \\quad \\forall K \in \\mathcal{T}\} " % (i, i))
    for i, f in enumerate(formdata.coefficients):
        strings.append("W_h^{%d} = \{v : v \\vert_K \in \\mathcal{Q}_{%d}(K) \\quad \\forall K \in \\mathcal{T}\} " % (i, i))
    if strings:
        latex += "Function spaces:\n"
        latex += make_align(strings)
    
    # Define basis functions and functions
    # TODO: Get names of arguments from form file
    strings = []
    for i,e in enumerate(formdata.basisfunctions):
        strings.append("v_h^{%d} \\in V_h^{%d} " % (i, i))
    for i,f in enumerate(formdata.coefficients):
        strings.append("w_h^{%d} \\in W_h^{%d} " % (i, i))
    if strings:
        latex += "Form arguments:\n"
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
    if strings:
        latex += "Variables:\n"
        latex += make_align(strings)
    
    # Join form arguments for "a(...) ="
    b = ", ".join("v_h^{%d}" % i for (i,v) in enumerate(formdata.basisfunctions))
    c = ", ".join("w_h^{%d}" % i for (i,w) in enumerate(formdata.coefficients))
    arguments = "; ".join((b, c))
    latex += "Form:\n"
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
    "Generate LaTeX code for a UFL expression or form."
    if isinstance(expression, Form):
        return form2latex(expression)
    basisfunction_renumbering = dict((f,f._count) for f in extract_basisfunctions(expression))
    coefficient_renumbering = dict((f,f._count) for f in extract_coefficients(expression))
    return expression2latex(expression, basisfunction_renumbering, coefficient_renumbering)

def forms2latexdocument(forms, uflfilename):
    "Generate a complete LaTeX document for a list of UFL forms."
    # Analyse validity of forms
    for k,v in forms:
        errors = validate_form(v)
        if errors:
            msg = "Found errors in form '%s':\n%s" % (k, errors)
            raise RuntimeError, msg
    
    # Define template for overall document
    latex = r"""\documentclass{article}
    \usepackage{amsmath}
    
    \begin{document}
    
    \section{UFL Forms from file %s}
    
    """ % uflfilename.replace("_", "\\_")
    
    # Generate latex code for each form
    for name, form in forms:
        l = ufl2latex(form)
        latex += "\\subsection{Form %s}\n" % name
        latex += l
    
    latex += r"""
    \end{document}
    """
    return latex

def write_file(filename, text):
    f = open(filename, "w")
    f.write(text)
    f.close()

def openpdf(pdffilename):
    # TODO: Add option for this. Is there a portable way to do this? like "open foo.pdf in pdf viewer"
    os.system("evince %s &" % pdffilename)

def uflfile2latex(uflfilename, latexfilename):
    "Compile a LaTeX file from a .ufl file."
    forms = load_forms(uflfilename)
    latex = forms2latexdocument(forms, uflfilename) 
    write_file(latexfilename, latex)

def latex2pdf(latexfilename, pdffilename):
    os.system("pdflatex %s %s" % (latexfilename, pdffilename)) # TODO: Use subprocess
    openpdf(pdffilename)

def uflfile2pdf(uflfilename, latexfilename, pdffilename):
    "Compile a .pdf file from a .ufl file."
    uflfile2latex(uflfilename, latexfilename)
    latex2pdf(latexfilename, pdffilename)

def codestructure2latex(code, formdata):
    "TODO: Document me"
    
    bfn = formdata.basisfunction_renumbering
    cfn = formdata.coefficient_renumbering
    
    def dep2latex(dep):
        # TODO: Better formatting of dependencies
        return "Dependencies:\n\\begin{verbatim}\n%s\n\\end{verbatim}" % str(dep)
    
    latex = ""
    newline = "\\\\\n"
    for dep, stack in code.stacks.iteritems():
        latex += "\n\n"
        latex += dep2latex(dep)
        latex += "\n\\begin{align}\n"
        
        for vinfo in stack[:-1]:
            vl = expression2latex(vinfo.variable, bfn, cfn)
            el = expression2latex(vinfo.variable._expression, bfn, cfn)
            latex += "%s &= %s %s" % (vl, el, newline)
        
        vinfo = stack[-1]
        vl = expression2latex(vinfo.variable, bfn, cfn)
        el = expression2latex(vinfo.variable._expression, bfn, cfn)
        latex += "%s &= %s" % (vl, el)
        
        latex += "\n\\end{align}\n"
    
    return latex

def codestructure2pdf(code, formdata, latexfilename, pdffilename):
    "Compile a .pdf file from a CodeStructure."
    
    # FIXME: Add final variable representing integrand
    # FIXME: Handle lists of named forms (code / formdata)
    # FIXME: Sort dependency sets in a sensible way (preclude to a good quadrature code generator)
    # FIXME: We're generating a lot of "Variable(Variable(...))" expressions!
    
    # Define template for overall document
    latex = r"""\documentclass{article}
    \usepackage{amsmath}
    
    \begin{document}
    
    \section{Code structure}
    
    """ # % uflfilename.replace("_", "\\_")

    latex += codestructure2latex(code, formdata)
    
    latex += r"""
    \end{document}
    """
    
    write_file(latexfilename, latex)
    latex2pdf(latexfilename, pdffilename)
