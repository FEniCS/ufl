"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-11-05"

# Modified by Anders Logg, 2008.

import os
from itertools import chain
from collections import defaultdict

from ..output import ufl_error, ufl_debug, ufl_warning
from ..common import UFLTypeDefaultDict, write_file, openpdf

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
from .latextools import align, document, verbatim

from .dependencies import DependencySet, CodeStructure, split_by_dependencies
from .variables import mark_duplications
from .transformations import expand_compounds


# --- Tools for LaTeX rendering of UFL expressions ---
# TODO: Precedence handling

def build_precedence_map():
    precedence_list = [] # FIXME: Review this list very carefully!

    precedence_list.append((Sum,))
    
    # FIXME: What to do with these?
    precedence_list.append((ListTensor, ComponentTensor))
    precedence_list.append((NegativeRestricted, PositiveRestricted))
    precedence_list.append((Conditional,))
    precedence_list.append((LE, GT, GE, NE, EQ, LT))
    
    precedence_list.append((Div, Grad, Curl, Rot, SpatialDerivative, VariableDerivative,
        Determinant, Trace, Cofactor, Inverse, Deviatoric))
    precedence_list.append((Product, Division, Cross, Dot, Outer, Inner))
    precedence_list.append((Indexed, Transposed, Power))
    precedence_list.append((Abs, Cos, Exp, Ln, Sin, Sqrt))
    precedence_list.append((Variable,))
    precedence_list.append((IntValue, FloatValue, ScalarValue, Zero, Identity,
        FacetNormal, Constant, VectorConstant, TensorConstant,
        BasisFunction, Function, MultiIndex))
    
    precedence_map = {}
    k = 0
    for p in precedence_list:
        for c in p:
            precedence_map[c] = k
        k += 1
    return precedence_map


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
    d[Zero]          = lambda x: "0" if not x.shape() else r"{\mathbf 0}"
    d[FloatValue]    = lambda x: "{%s}" % x._value
    d[IntValue]      = lambda x: "{%s}" % x._value
    d[FacetNormal]   = lambda x: "n"
    d[Identity]      = lambda x: "I"
    d[BasisFunction] = lambda x: "{v_h^{%d}}" % basisfunction_renumbering[x] # Using ^ for function numbering and _ for indexing
    d[Function]      = lambda x: "{w_h^{%d}}" % coefficient_renumbering[x]
    d[Constant]      = lambda x: "{w_h^{%d}}" % coefficient_renumbering[x]
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

def expression2latex(expression, basisfunction_renumbering, coefficient_renumbering):
    handlers = latex_handlers(basisfunction_renumbering, coefficient_renumbering)
    latex = transform(expression, handlers)
    return latex

def element2latex(element):
    return "{\mbox{%s}}" % str(element)

domain_strings = { "cell": "\\Omega", "exterior_facet": "\\Gamma^{ext}", "interior_facet": "\\Gamma^{int}" }
dx_strings = { "cell": "dx", "exterior_facet": "ds", "interior_facet": "dS" }

def form2latex(form, formname="a", basisfunction_names = {}, function_names = {}):
    formdata = FormData(form)
    sections = []
    
    # Define elements
    lines = []
    for i, f in enumerate(formdata.basisfunctions):
        lines.append("\\mathcal{P}_{%d} = \{%s\} " % (i, element2latex(f.element())))
    for i, f in enumerate(formdata.coefficients):
        lines.append("\\mathcal{Q}_{%d} = \{%s\} " % (i, element2latex(f.element())))
    if lines:
        sections.append(("Finite elements", align(lines)))
    
    # Define function spaces
    lines = []
    for i, f in enumerate(formdata.basisfunctions):
        lines.append("V_h^{%d} = \{v : v \\vert_K \in \\mathcal{P}_{%d}(K) \\quad \\forall K \in \\mathcal{T}\} " % (i, i))
    for i, f in enumerate(formdata.coefficients):
        lines.append("W_h^{%d} = \{v : v \\vert_K \in \\mathcal{Q}_{%d}(K) \\quad \\forall K \in \\mathcal{T}\} " % (i, i))
    if lines:
        sections.append(("Function spaces", align(lines)))
    
    # Define basis functions and functions
    lines = []
    for i,e in enumerate(formdata.basisfunctions):
        name = basisfunction_names.get(f, None)
        name = "" if name is None else (name + " = ")
        lines.append("%sv_h^{%d} \\in V_h^{%d} " % (name, i, i))
    for i,f in enumerate(formdata.coefficients):
        name = function_names.get(f, None)
        name = "" if name is None else (name + " = ")
        lines.append("%sw_h^{%d} \\in W_h^{%d} " % (name, i, i))
    if lines:
        sections.append(("Form arguments", align(lines)))
    
    # Define variables
    handled_variables = set()
    integrals = list(chain(form.cell_integrals(),
                           form.exterior_facet_integrals(),
                           form.interior_facet_integrals()))
    lines = []
    for itg in integrals:
        vars = extract_variables(itg.integrand())
        for v in vars:
            if not v._count in handled_variables:
                handled_variables.add(v._count)
                exprlatex = expression2latex(v._expression, formdata.basisfunction_renumbering, formdata.coefficient_renumbering)
                lines.append(("s_{%d}" % v._count, "= %s" % exprlatex))
    if lines:
        sections.append(("Variables", align(lines)))
    
    # Join form arguments for signature "a(...) ="
    b = ", ".join("v_h^{%d}" % i for (i,v) in enumerate(formdata.basisfunctions))
    c = ", ".join("w_h^{%d}" % i for (i,w) in enumerate(formdata.coefficients))
    arguments = "; ".join((b, c))
    signature = "%s(%s) = " % (formname, arguments, )
    
    # Define form as sum of integrals
    lines = []
    a = signature; p = ""
    for itg in integrals:
        # TODO: Get list of expression strings instead of single expression!
        integrand_string = expression2latex(itg.integrand(), formdata.basisfunction_renumbering, formdata.coefficient_renumbering)
        b = p + "\\int_{%s_%d}" % (domain_strings[itg._domain_type], itg._domain_id)
        c = "\\left[ { %s } \\right] \,%s" % (integrand_string, dx_strings[itg._domain_type])
        lines.append((a, b, c))
        a = "{}"; p = "{}+ "
    sections.append(("Form", align(lines)))
    
    return sections

def ufl2latex(expression):
    "Generate LaTeX code for a UFL expression or form (wrapper for form2latex and expression2latex)."
    if isinstance(expression, Form):
        return form2latex(expression)
    basisfunction_renumbering = dict((f,f._count) for f in extract_basisfunctions(expression))
    coefficient_renumbering = dict((f,f._count) for f in extract_coefficients(expression))
    return expression2latex(expression, basisfunction_renumbering, coefficient_renumbering)

# --- LaTeX rendering of composite UFL objects ---

def bfname(i):
    return "{v_h^%d}" % i

def cfname(i):
    return "{w_h^%d}" % i

def dep2latex(dep):
    deps = []
    if dep.cell:
        deps.append("K") # TODO: Better symbol
    if dep.mapping:
        deps.append("G") # TODO: Better symbol
    if dep.facet:
        deps.append("n") # TODO: Better symbol
    if dep.coordinates:
        deps.append("x") # TODO: Better symbol
    for i,v in enumerate(dep.basisfunctions):
        if v: deps.append(bfname(i))
    for i,w in enumerate(dep.functions):
        if w: deps.append(cfname(i))
    return "Dependencies: ${ %s }$." % ", ".join(deps)

def code2latex(vinfo, code, formdata):
    "TODO: Document me"
    bfn = formdata.basisfunction_renumbering
    cfn = formdata.coefficient_renumbering

    deplist = code.stacks.keys()
    # FIXME: Sort dependency sets in a sensible way (preclude to a good quadrature code generator)
    deplist = sorted(deplist)
    
    pieces = []
    for dep in deplist:
        stack = code.stacks[dep]
        
        lines = []
        for vinfo in stack[:-1]:
            vl = expression2latex(vinfo.variable, bfn, cfn)
            el = expression2latex(vinfo.variable._expression, bfn, cfn)
            lines.append((vl, "= " + el))
        
        vinfo = stack[-1]
        vl = expression2latex(vinfo.variable, bfn, cfn)
        el = expression2latex(vinfo.variable._expression, bfn, cfn)
        lines.append((vl, "= " + el))
        
        pieces.append("\n")
        pieces.append(dep2latex(dep))
        pieces.append(align(lines))
    
    # Add final variable representing integrand
    pieces.append("\n")
    pieces.append("Variable representing integrand. " + dep2latex(vinfo.deps))
    vl = expression2latex(vinfo.variable, bfn, cfn)
    el = expression2latex(vinfo.variable._expression, bfn, cfn)
    lines = [(vl, "= " + el)]
    pieces.append(align(lines))
    
    # Could also return list of (title, body) parts for subsections if wanted
    body = "\n".join(pieces)
    return body

def integrand2code(integrand, formdata, basisfunction_deps, function_deps):
    # Try to pick up duplications on the most abstract level
    integrand = mark_duplications(integrand)
    
    # Expand grad, div, inner etc to index notation
    integrand = expand_compounds(integrand, formdata.geometric_dimension)
    
    # Try to pick up duplications on the index notation level
    integrand = mark_duplications(integrand)
    
    # FIXME: Apply AD stuff for Diff and propagation of SpatialDerivative to Terminal nodes. Or do we need to build code structure first to do this better?
    #integrand = FIXME(integrand)
    #integrand = compute_diffs(integrand)
    #integrand = propagate_spatial_diffs(integrand)
    
    # Try to pick up duplications after propagating derivatives
    #integrand = mark_duplications(integrand)
    
    (vinfo, code) = split_by_dependencies(integrand, formdata, basisfunction_deps, function_deps)
    
    return vinfo, code

def formdata2latex(formdata):
    return verbatim(str(formdata)) # TODO

def form2code2latex(form):
    formdata = FormData(form)
        
    # Define toy input to split_by_dependencies
    basisfunction_deps = []
    fs = (False,)*formdata.num_coefficients
    for i in range(formdata.rank):
        bfs = tuple(i == j for j in range(formdata.rank)) # TODO: More dependencies depending on element
        d = DependencySet(bfs, fs)
        basisfunction_deps.append(d)
    
    function_deps = []
    bfs = (False,)*formdata.rank
    for i in range(formdata.num_coefficients):
        fs = tuple(i == j for j in range(formdata.num_coefficients)) # TODO: More dependencies depending on element
        d = DependencySet(bfs, fs)
        function_deps.append(d)
    
    title = "Form data"
    body = formdata2latex(formdata)
    sections = [(title, body)]
    
    for itg in form.cell_integrals():
        vinfo, itgcode = integrand2code(itg.integrand(), formdata, basisfunction_deps, function_deps)
        title = "%s integral over domain %d" % (itg.domain_type(), itg.domain_id())
        body = code2latex(vinfo, itgcode, formdata)
        sections.append((title, body))
    
    return sections

# --- Creating complete documents ---

def forms2latexdocument(forms, uflfilename, compile=False):
    "Render forms from a .ufl file as a LaTeX document."
    sections = []
    for name, form in forms:
        title = "Form %s" % name
        if compile:
            body = form2code2latex(form)
        else:
            body = form2latex(form) #, formname, basisfunction_names, function_names) # TODO
        sections.append((title, body))

    if compile:
        title = "Compiled forms from UFL file %s" % uflfilename.replace("_", "\\_")
    else:
        title = "Forms from UFL file %s" % uflfilename.replace("_", "\\_")
    return document(title, sections)

# --- File operations ---

def ufl2tex(uflfilename, latexfilename, compile=False):
    "Compile a .tex file from a .ufl file."
    forms = load_forms(uflfilename) # TODO: Get function names from this
    latex = forms2latexdocument(forms, uflfilename, compile) 
    write_file(latexfilename, latex)

def tex2pdf(latexfilename, pdffilename):
    # TODO: Use subprocess.
    # TODO: Options for this.
    os.system("pdflatex %s %s" % (latexfilename, pdffilename))
    openpdf(pdffilename)

def ufl2pdf(uflfilename, latexfilename, pdffilename, compile=False):
    "Compile a .pdf file from a .ufl file."
    ufl2tex(uflfilename, latexfilename, compile)
    tex2pdf(latexfilename, pdffilename)
