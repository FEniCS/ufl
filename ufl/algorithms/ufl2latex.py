"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-11-27"

# Modified by Anders Logg, 2008.

import os
from itertools import chain

from ufl.output import ufl_error, ufl_assert, ufl_warning
from ufl.common import write_file, openpdf
from ufl.permutation import compute_indices

# All classes:
from ufl.zero import Zero
from ufl.scalar import ScalarValue, FloatValue, IntValue, ScalarSomething
from ufl.variable import Variable
from ufl.basisfunction import BasisFunction
from ufl.function import Function, Constant, VectorConstant, TensorConstant
from ufl.geometry import FacetNormal
from ufl.indexing import MultiIndex, Indexed, Index, FixedIndex, AxisType
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ufl.form import Form

# Other algorithms:
from ufl.algorithms.analysis import extract_basisfunctions, extract_coefficients, extract_variables
from ufl.algorithms.formdata import FormData
from ufl.algorithms.formfiles import load_forms
from ufl.algorithms.latextools import align, document, verbatim

from ufl.algorithms.dependencies import DependencySet, split_by_dependencies
from ufl.algorithms.transformations import expand_compounds, mark_duplications, Transformer


# --- Tools for LaTeX rendering of UFL expressions ---

# TODO: Finish precedence mapping
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

# Utility for parentesizing string (TODO: Finish precedence handling):
def par(s, condition=True):
    if condition:
        return "\\left(%s\\right)" % s
    return str(s)
    
def format_index(ii):
    if isinstance(ii, FixedIndex):
        s = "%d" % ii._value
    elif isinstance(ii, Index):
        s = "i_{%d}" % ii._count
    elif isinstance(ii, AxisType):
        s = ":" # TODO: How to express this in natural syntax?
    else:
        ufl_error("Unknown index type %s." % type(ii))
    return s

def format_multi_index(ii, formatstring="%s"):
    return "".join(formatstring % format_index(i) for i in ii)

# TODO: Handle parantesizing based on precedence
# TODO: Handle line wrapping
# TODO: Handle ListTensors of rank > 1 correctly
class Expression2LatexHandler(Transformer):
    def __init__(self, basisfunction_renumbering, coefficient_renumbering):
        Transformer.__init__(self)
        self.basisfunction_renumbering = basisfunction_renumbering
        self.coefficient_renumbering = coefficient_renumbering
    
    # --- Terminal objects ---
    
    def scalar_value(self, o):
        return "{%s}" % o._value
    
    def zero(self, o):
        return "0" if not o.shape() else r"{\mathbf 0}"
    
    def identity(self, o):
        return "I"
    
    def facet_normal(self, o):
        return "n"
    
    def basis_function(self, o):
        return "{v_h^{%d}}" % self.basisfunction_renumbering[o] # Using ^ for function numbering and _ for indexing
    
    def function(self, o):
        return "{w_h^{%d}}" % self.coefficient_renumbering[o]
    
    def constant(self, o):
        return "{w_h^{%d}}" % self.coefficient_renumbering[o]
    
    def multi_index(self, o):
        return format_multi_index(o, formatstring="{%s}")
    
    def variable(self, o):
        # TODO: Ensure variable has been handled
        return "s_{%d}" % o._count
    
    # --- Non-terminal objects ---
    
    def sum(self, o, *ops):
        return " + ".join(par(op) for op in ops)
    
    def product(self, o, *ops):
        return " ".join(par(op) for op in ops)
    
    def division(self, o, a, b):
        return r"\frac{%s}{%s}" % (a, b)
    
    def abs(self, o, a):
        return "|%s|" % a
    
    def transposed(self, o, a):
        return "{%s}^T" % par(a)
    
    def indexed(self, o, a, b):
        return "{%s}_{%s}" % (par(a), b)
    
    def spatial_derivative(self, o, f, ii):
        ii = o.operands()[1]
        n = len(ii)
        y = format_multi_index(ii, formatstring="\partial{}x_{%s}")
        l = r"\left["
        r = r"\right]"
        d = "" if (n == 1) else (r"^%d" % n)
        nom = r"\partial%s%s%s%s" % (l, f, r, d)
        denom = r"%s" % y
        return r"\frac{%s}{%s}" % (nom, denom)
    
    def variable_derivative(self, o, f, v):
        nom   = r"\partial\left[%s\right]" % f
        denom = r"\partial\left[%s\right]" % v
        return r"\frac{%s}{%s}" % (nom, denom)
    
    def grad(self, o, f):
        return "\\nabla{%s}" % par(f)
    
    def div(self, o, f):
        return "\\nabla{\\cdot %s}" % par(f)
    
    def curl(self, o, f):
        return "\\nabla{\\times %s}" % par(f)
    
    def rot(self, o, f):
        return "\\rot{%s}" % par(f)
    
    def sqrt(self, o, f):
        return "%s^{\frac 1 2}" % par(f)
    
    def exp(self, o, f):
        return "e^{%s}" % par(f)
    
    def ln(self, o, f):
        return "\\ln{%s}" % par(f)
    
    def cos(self, o, f):
        return "\\cos{%s}" % par(f)
    
    def sin(self, o, f):
        return "\\sin{%s}" % par(f)
    
    def power(self, o, a, b):
        return "{%s}^{%s}" % (par(a), par(b))
    
    def outer(self, o, a, b):
        return "{%s}\\otimes{%s}" % (par(a), par(b))
    
    def inner(self, o, a, b):
        return "{%s}:{%s}" % (par(a), par(b))
    
    def dot(self, o, a, b):
        return "{%s}\\cdot{%s}" % (par(a), par(b))
    
    def cross(self, o, a, b):
        return "{%s}\\times{%s}" % (par(a), par(b))
    
    def trace(self, o, A):
        return "tr{%s}" % par(A)
    
    def determinant(self, o, A):
        return "det{%s}" % par(A)
    
    def inverse(self, o, A):
        return "{%s}^{-1}" % par(A)
    
    def deviatoric(self, o, A):
        return "dev{%s}" % par(A)
    
    def cofactor(self, o, A):
        return "cofac{%s}" % par(A)
    
    def list_tensor(self, o):
        shape = o.shape()
        if len(shape) == 1:
            ops = [self.visit(op) for op in o.operands()]
            l = " \\\\ \n ".join(ops)
        elif len(shape) == 2:
            rows = []
            for row in o.operands():
                cols = (self.visit(op) for op in row.operands())
                rows.append( " & \n ".join(cols) )
            l = " \\\\ \n ".join(rows)
        else:
            ufl_error("TODO: LaTeX handler for list tensor of rank 3 or higher not implemented!")
        return "\\left[\\begin{matrix}{%s}\\end{matrix}\\right]^T" % l
    
    def component_tensor(self, o, *ops):
        A, ii = ops
        return "\\left[A \\quad | \\quad A_{%s} = {%s} \\quad \\forall {%s} \\right]" % (ii, A, ii)
    
    def positive_restricted(self, o, f):
        return "{%s}^+" % par(f)
    
    def negative_restricted(self, o, f):
        return "{%s}^-" % par(f)
    
    def conditional(self, o, a, b):
        return "(%s %s %s)" % (a, o._name, b)
    
    def eq(self, o, a, b):
        return "(%s = %s)" % (a, b)
    
    def ne(self, o, a, b):
        return "(%s \\ne %s)" % (a, b)
    
    def le(self, o, a, b):
        return "(%s \\le %s)" % (a, b)
    
    def ge(self, o, a, b):
        return "(%s \\ge %s)" % (a, b)
    
    def lt(self, o, a, b):
        return "(%s < %s)" % (a, b)
    
    def gt(self, o, a, b):
        return "(%s > %s)" % (a, b)
    
    def conditional(self, o, c, t, f):
        l = "\\begin{cases}\n"
        l += "%s, &\text{if }\quad %s, \\\\\n" % (t, c)
        l += "%s, &\text{otherwise.}\n" % f
        l += "\\end{cases}"
        return l
    
    def expr(self, o, *ops):
        ufl_error("Missing handler for type %s" % str(type(o)))
    
def expression2latex(expression, basisfunction_renumbering, coefficient_renumbering):
    visitor = Expression2LatexHandler(basisfunction_renumbering, coefficient_renumbering)
    return visitor.visit(expression)

def element2latex(element):
    return "{\mbox{%s}}" % str(element)

domain_strings = { "cell": "\\Omega", "exterior_facet": "\\Gamma^{ext}", "interior_facet": "\\Gamma^{int}" }
dx_strings = { "cell": "dx", "exterior_facet": "ds", "interior_facet": "dS" }

def form2latex(form, formname="a", basisfunction_names = None, function_names = None):
    if basisfunction_names is None: basisfunction_names = {}
    if function_names is None: function_names = {}
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
    for i, f in enumerate(formdata.basisfunctions):
        name = basisfunction_names.get(f, None)
        name = "" if name is None else (name + " = ")
        lines.append("%sv_h^{%d} \\in V_h^{%d} " % (name, i, i))
    for i, f in enumerate(formdata.coefficients):
        name = function_names.get(f, None)
        name = "" if name is None else (name + " = ")
        lines.append("%sw_h^{%d} \\in W_h^{%d} " % (name, i, i))
    if lines:
        sections.append(("Form arguments", align(lines)))
    
    # TODO: Wrap ListTensors, ComponentTensor and Conditionals in expression as variables before transformation
    
    # Define variables
    handled_variables = set()
    integrals = list(chain(form.cell_integrals(),
                           form.exterior_facet_integrals(),
                           form.interior_facet_integrals()))
    lines = []
    for itg in integrals:
        variables = extract_variables(itg.integrand())
        for v in variables:
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
    if dep.runtime:
        deps.append("K")
    if dep.coordinates:
        deps.append("x")
    for i,v in enumerate(dep.basisfunctions):
        if v: deps.append(bfname(i))
    return "Dependencies: ${ %s }$." % ", ".join(deps)

def dependency_sorting(deplist, rank): # TODO: Use this in SFC
    def split(deps, state):
        left = []
        todo = []
        for dep in deps:
            if state.covers(dep):
                todo.append(dep)
            else:
                left.append(dep)
        return todo, left
    
    deplistlist = []
    state = DependencySet((False,)*rank)
    left = deplist
    
    # --- Initialization time
    state.runtime = False
    
    state.coordinates = False
    precompute, left = split(left, state)
    deplistlist.append(precompute)
    
    state.coordinates = True
    precompute_quad, left = split(left, state)
    deplistlist.append(precompute_quad)
    
    indices = compute_indices((2,)*rank)
    for bfs in indices[1:]: # skip (0,...,0)
        state.basisfunctions = map(bool, reversed(bfs))
        next, left = split(left, state)
        deplistlist.append(next)
    
    # --- Runtime
    state.runtime = True
    
    state.coordinates = False
    runtime, left = split(left, state)
    deplistlist.append(runtime)
    
    state.coordinates = True
    runtime_quad, left = split(left, state)
    deplistlist.append(runtime_quad)
    
    indices = compute_indices((2,)*rank)
    for bfs in indices[1:]: # skip (0,...,0)
        state.basisfunctions = map(bool, reversed(bfs))
        next, left = split(left, state)
        deplistlist.append(next)
    
    ufl_assert(not left, "Shouldn't have anything left!")
    
    print
    print "Created deplistlist:"
    for deps in deplistlist:
        print
        print "--- new stage:"
        print "\n".join(map(str, deps))
    print

    return deplistlist

def code2latex(integrand_vinfo, code, formdata):
    "TODO: Document me"
    bfn = formdata.basisfunction_renumbering
    cfn = formdata.coefficient_renumbering

    # Sort dependency sets in a sensible way (preclude to a good quadrature code generator)
    #deplistlist = [sorted(code.stacks.keys())]
    deplistlist = dependency_sorting(code.stacks.keys(), len(bfn))
    
    pieces = []
    for deplist in deplistlist:
        pieces.append("\n\n(Debugging: getting next list of dependencies)")
        for dep in deplist:
            lines = []
            for vinfo in code.stacks[dep]:
                vl = expression2latex(vinfo.variable, bfn, cfn)
                el = expression2latex(vinfo.variable._expression, bfn, cfn)
                lines.append((vl, "= " + el))
            pieces.extend(("\n", dep2latex(dep), align(lines)))
    
    # Add final variable representing integrand
    vl = expression2latex(integrand_vinfo.variable, bfn, cfn)
    el = expression2latex(integrand_vinfo.variable._expression, bfn, cfn)
    pieces.append("\n")
    pieces.append("Variable representing integrand. " + dep2latex(integrand_vinfo.deps))
    pieces.append(align([(vl, "= " + el)]))
    
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
    for i in range(formdata.rank):
        bfs = tuple(i == j for j in range(formdata.rank)) 
        d = DependencySet(bfs, coordinates=True) # TODO: Toggle coordinates depending on element
        basisfunction_deps.append(d)
    
    function_deps = []
    bfs = (False,)*formdata.rank
    for i in range(formdata.num_coefficients):
        d = DependencySet(bfs, runtime=True, coordinates=True) # TODO: Toggle coordinates depending on element
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
    flags = " -file-line-error-style -interaction=nonstopmode"
    os.system("pdflatex %s %s %s" % (flags, latexfilename, pdffilename))
    openpdf(pdffilename)

def ufl2pdf(uflfilename, latexfilename, pdffilename, compile=False):
    "Compile a .pdf file from a .ufl file."
    ufl2tex(uflfilename, latexfilename, compile)
    tex2pdf(latexfilename, pdffilename)
