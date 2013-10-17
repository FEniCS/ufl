"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008-2009.
# Modified by Kristian B. Oelgaard, 2011
#
# First added:  2008-05-07
# Last changed: 2011-10-21

from itertools import chain

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import write_file, pdflatex, openpdf
from ufl.permutation import compute_indices

# All classes:
from ufl.variable import Variable
from ufl.indexing import Index, FixedIndex
from ufl.indexed import Indexed
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.indexsum import IndexSum
from ufl.tensoralgebra import Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ufl.mathfunctions import Sqrt, Exp, Ln, Cos, Sin, Tan, Cosh, Sinh, Tanh, Acos, Asin, Atan, Atan2, Erf, BesselJ, BesselY, BesselI, BesselK
from ufl.restriction import PositiveRestricted, NegativeRestricted, CellAvg, FacetAvg
from ufl.differentiation import VariableDerivative, Grad, Div, Curl, NablaGrad, NablaDiv
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ufl.form import Form
from ufl.integral import Measure
from ufl.classes import terminal_classes
from ufl.domains import Domain, Region

# Other algorithms:
from ufl.algorithms.preprocess import preprocess
from ufl.algorithms.analysis import extract_variables
from ufl.algorithms.formfiles import load_forms
from ufl.algorithms.latextools import align, document, verbatim

from ufl.algorithms.transformer import Transformer
from ufl.algorithms.graph import build_graph, partition, extract_outgoing_vertex_connections


# TODO: Maybe this can be cleaner written using the graph utilities


# --- Tools for LaTeX rendering of UFL expressions ---

# TODO: Finish precedence mapping
def build_precedence_map():
    precedence_list = [] # TODO: Review this list very carefully!

    precedence_list.append((Sum,))
    precedence_list.append((IndexSum,))

    # TODO: What to do with these?
    precedence_list.append((ListTensor, ComponentTensor))
    precedence_list.append((CellAvg, ))
    precedence_list.append((NegativeRestricted, PositiveRestricted))
    precedence_list.append((Conditional,))
    precedence_list.append((LE, GT, GE, NE, EQ, LT))

    precedence_list.append((Div, Grad, NablaGrad, NablaDiv, Curl, VariableDerivative,
                            Determinant, Trace, Cofactor, Inverse, Deviatoric))
    precedence_list.append((Product, Division, Cross, Dot, Outer, Inner))
    precedence_list.append((Indexed, Transposed, Power))
    precedence_list.append((Abs, Cos, Cosh, Exp, Ln, Sin, Sinh, Sqrt, Tan, Tanh, Acos, Asin, Atan, Atan2, Erf, BesselJ, BesselY, BesselI, BesselK))
    precedence_list.append((Variable,))
    precedence_list.append(terminal_classes)

    precedence_map = {}
    k = 0
    for p in precedence_list:
        for c in p:
            precedence_map[c] = k
        k += 1
    return precedence_map

# Utility for parentesizing string
def par(s, condition=True): # TODO: Finish precedence handling by adding condition argument to calls to this function!
    if condition:
        return r"\left(%s\right)" % s
    return str(s)

def format_index(ii):
    if isinstance(ii, FixedIndex):
        s = "%d" % ii._value
    elif isinstance(ii, Index):
        s = "i_{%d}" % ii._count
    else:
        error("Invalid index type %s." % type(ii))
    return s

def format_multi_index(ii, formatstring="%s"):
    return "".join(formatstring % format_index(i) for i in ii)

def bfname(i):
    return "{v_h^%d}" % i

def cfname(i):
    return "{w_h^%d}" % i

# TODO: Handle line wrapping
# TODO: Handle ListTensors of rank > 1 correctly
class Expression2LatexHandler(Transformer):
    def __init__(self, argument_names=None, coefficient_names=None):
        Transformer.__init__(self)
        self.argument_names = argument_names
        self.coefficient_names = coefficient_names

    # --- Terminal objects ---

    def scalar_value(self, o):
        if o.shape():
            return r"{\mathbf %s}" % o._value
        return "{%s}" % o._value

    def zero(self, o):
        return "0" if not o.shape() else r"{\mathbf 0}"

    def identity(self, o):
        return r"{\mathbf I}"

    def permutation_symbol(self, o):
        return r"{\mathbf \varepsilon}"

    def facet_normal(self, o):
        return r"{\mathbf n}"

    def argument(self, o):
        # Using ^ for argument numbering and _ for indexing since indexing is more common than exponentiation
        if self.argument_names is None:
            return bfname(o.count())
        return self.argument_names[o.count()]

    def coefficient(self, o):
        # Using ^ for coefficient numbering and _ for indexing since indexing is more common than exponentiation
        if self.coefficient_names is None:
            return cfname(o.count())
        return self.coefficient_names[o.count()]
    constant = coefficient

    def multi_index(self, o):
        return format_multi_index(o, formatstring="{%s}")

    def variable(self, o):
        # TODO: Ensure variable has been handled
        e, l = o.operands()
        return "s_{%d}" % l._count

    # --- Non-terminal objects ---

    def index_sum(self, o, f, i):
        return r"\sum_{%s}%s" % (i, par(f))

    def sum(self, o, *ops):
        return " + ".join(par(op) for op in ops)

    def product(self, o, *ops):
        return " ".join(par(op) for op in ops)

    def division(self, o, a, b):
        return r"\frac{%s}{%s}" % (a, b)

    def abs(self, o, a):
        return r"\|%s\|" % a

    def transposed(self, o, a):
        return "{%s}^T" % par(a)

    def indexed(self, o, a, b):
        return "{%s}_{%s}" % (par(a), b)

    def variable_derivative(self, o, f, v):
        nom   = r"\partial%s" % par(f)
        denom = r"\partial%s" % par(v)
        return r"\frac{%s}{%s}" % (nom, denom)

    def coefficient_derivative(self, o, f, w, v):
        nom   = r"\partial%s" % par(f)
        denom = r"\partial%s" % par(w)
        return r"\frac{%s}{%s}[%s]" % (nom, denom, v) # TODO: Fix this syntax...

    def grad(self, o, f):
        return r"\mathbf{grad}{%s}" % par(f)

    def div(self, o, f):
        return r"\mathbf{grad}{%s}" % par(f)

    def nabla_grad(self, o, f):
        return r"\nabla{\otimes %s}" % par(f)

    def nabla_div(self, o, f):
        return r"\nabla{\cdot %s}" % par(f)

    def curl(self, o, f):
        return r"\nabla{\times %s}" % par(f)

    def sqrt(self, o, f):
        return r"%s^{\frac 1 2}" % par(f)

    def exp(self, o, f):
        return "e^{%s}" % f

    def ln(self, o, f):
        return r"\ln{%s}" % par(f)

    def cos(self, o, f):
        return r"\cos{%s}" % par(f)

    def sin(self, o, f):
        return r"\sin{%s}" % par(f)

    def tan(self, o, f):
        return r"\tan{%s}" % par(f)

    def cosh(self, o, f):
        return r"\cosh{%s}" % par(f)

    def sinh(self, o, f):
        return r"\sinh{%s}" % par(f)

    def tanh(self, o, f):
        return r"\tanh{%s}" % par(f)

    def acos(self, o, f):
        return r"\arccos{%s}" % par(f)

    def asin(self, o, f):
        return r"\arcsin{%s}" % par(f)

    def atan(self, o, f):
        return r"\arctan{%s}" % par(f)

    def atan2(self, o, f1, f2):
        return r"\arctan_2{%s,%s}" % (par(f1), par(f2))

    def erf(self, o, f):
        return r"\erf{%s}" % par(f)

    def bessel_j(self, o, nu, f):
        return r"J_{%s}{%s}" % (nu, par(f))

    def bessel_y(self, o, nu, f):
        return r"Y_{%s}{%s}" % (nu, par(f))

    def bessel_i(self, o, nu, f):
        return r"I_{%s}{%s}" % (nu, par(f))

    def bessel_K(self, o, nu, f):
        return r"K_{%s}{%s}" % (nu, par(f))

    def power(self, o, a, b):
        return "{%s}^{%s}" % (par(a), par(b))

    def outer(self, o, a, b):
        return r"{%s}\otimes{%s}" % (par(a), par(b))

    def inner(self, o, a, b):
        return "{%s}:{%s}" % (par(a), par(b))

    def dot(self, o, a, b):
        return r"{%s}\cdot{%s}" % (par(a), par(b))

    def cross(self, o, a, b):
        return r"{%s}\times{%s}" % (par(a), par(b))

    def trace(self, o, A):
        return "tr{%s}" % par(A) # TODO: Get built-in function syntax like \sin for this

    def determinant(self, o, A):
        return "det{%s}" % par(A) # TODO: Get built-in function syntax like \sin for this

    def inverse(self, o, A):
        return "{%s}^{-1}" % par(A)

    def deviatoric(self, o, A):
        return "dev{%s}" % par(A) # TODO: Get built-in function syntax like \sin for this

    def cofactor(self, o, A):
        return "cofac{%s}" % par(A) # TODO: Get built-in function syntax like \sin for this

    def skew(self, o, A):
        return "skew{%s}" % par(A) # TODO: Get built-in function syntax like \sin for this

    def sym(self, o, A):
        return "sym{%s}" % par(A) # TODO: Get built-in function syntax like \sin for this

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
            error("TODO: LaTeX handler for list tensor of rank 3 or higher not implemented!")
        return "\\left[\\begin{matrix}{%s}\\end{matrix}\\right]^T" % l

    def component_tensor(self, o, *ops):
        A, ii = ops
        return "\\left[A \\quad | \\quad A_{%s} = {%s} \\quad \\forall {%s} \\right]" % (ii, A, ii)

    def positive_restricted(self, o, f):
        return "{%s}^+" % par(f)

    def negative_restricted(self, o, f):
        return "{%s}^-" % par(f)

    def cell_avg(self, o, f):
        return "{%s}_K" % par(f)

    def eq(self, o, a, b):
        return "(%s = %s)" % (a, b)

    def ne(self, o, a, b):
        return r"(%s \ne %s)" % (a, b)

    def le(self, o, a, b):
        return r"(%s \le %s)" % (a, b)

    def ge(self, o, a, b):
        return r"(%s \ge %s)" % (a, b)

    def lt(self, o, a, b):
        return "(%s < %s)" % (a, b)

    def gt(self, o, a, b):
        return "(%s > %s)" % (a, b)

    def and_condition(self, o, a, b):
        return "(%s && %s)" % (a, b)

    def or_condition(self, o, a, b):
        return "(%s || %s)" % (a, b)

    def not_condition(self, o, a):
        return "!(%s)" % (a,)

    def conditional(self, o, c, t, f):
        l = "\\begin{cases}\n"
        l += "%s, &\text{if }\quad %s, \\\\\n" % (t, c)
        l += "%s, &\text{otherwise.}\n" % f
        l += "\\end{cases}"
        return l

    def expr(self, o):
        error("Missing handler for type %s" % str(type(o)))

def expression2latex(expression, argument_names=None, coefficient_names=None):
    visitor = Expression2LatexHandler(argument_names, coefficient_names)
    return visitor.visit(expression)

def element2latex(element):
    e = str(element)
    e = e.replace("<", "")
    e = e.replace(">", "")
    return r"{\mbox{%s}}" % e

domain_strings = { Measure.CELL: r"\Omega",
                   Measure.EXTERIOR_FACET: r"\Gamma^{ext}",
                   Measure.INTERIOR_FACET: r"\Gamma^{int}",
                   Measure.POINT: r"\Gamma^{point}",
                   Measure.MACRO_CELL: r"\Omega^{macro}",
                   Measure.SURFACE: r"\Gamma^{surface}",
                 }
default_domain_string = "d(?)"
dx_strings = { Measure.CELL: "dx",
               Measure.EXTERIOR_FACET: "ds",
               Measure.INTERIOR_FACET: "dS",
               Measure.POINT: "dP",
               Measure.MACRO_CELL: "dE",
               Measure.SURFACE: "dc",
             }

def form2latex(form, formdata):

    formname = formdata.name
    argument_names = formdata.argument_names
    coefficient_names = formdata.coefficient_names

    # List of sections to make latex document from
    sections = []

    # Define elements
    lines = []
    for i, f in enumerate(formdata.original_arguments):
        lines.append(r"\mathcal{P}_{%d} = \{%s\} " % (i, element2latex(f.element())))
    for i, f in enumerate(formdata.original_coefficients):
        lines.append(r"\mathcal{Q}_{%d} = \{%s\} " % (i, element2latex(f.element())))
    if lines:
        sections.append(("Finite elements", align(lines)))

    # Define function spaces
    lines = []
    for i, f in enumerate(formdata.original_arguments):
        lines.append("V_h^{%d} = \\{v : v \\vert_K \\in \\mathcal{P}_{%d}(K) \\quad \\forall K \\in \\mathcal{T}\\} " % (i, i))
    for i, f in enumerate(formdata.original_coefficients):
        lines.append("W_h^{%d} = \\{v : v \\vert_K \\in \\mathcal{Q}_{%d}(K) \\quad \\forall K \\in \\mathcal{T}\\} " % (i, i))
    if lines:
        sections.append(("Function spaces", align(lines)))

    # Define arguments and coefficients
    lines = []
    for i, f in enumerate(formdata.original_arguments):
        lines.append("%s = %s \\in V_h^{%d} " % (argument_names[i], bfname(i), i))
    for i, f in enumerate(formdata.original_coefficients):
        lines.append("%s = %s \\in W_h^{%d} " % (coefficient_names[i], cfname(i), i))
    if lines:
        sections.append(("Form arguments", align(lines)))

    # TODO: Wrap ListTensors, ComponentTensor and Conditionals in expression as variables before transformation

    # Define variables
    handled_variables = set()
    integrals = list(chain(form.integrals(Measure.CELL),
                           form.integrals(Measure.EXTERIOR_FACET),
                           form.integrals(Measure.INTERIOR_FACET),
                           form.integrals(Measure.POINT),
                           form.integrals(Measure.MACRO_CELL),
                           form.integrals(Measure.SURFACE)))
    ufl_assert(len(integrals) == len(form.integrals()),
               "Not handling non-standard integral types!")
    lines = []
    for itg in integrals:
        variables = extract_variables(itg.integrand())
        for v in variables:
            l = v._label
            if not l in handled_variables:
                handled_variables.add(l)
                exprlatex = expression2latex(v._expression, formdata.argument_names, formdata.coefficient_names)
                lines.append(("s_{%d}" % l._count, "= %s" % exprlatex))
    if lines:
        sections.append(("Variables", align(lines)))

    # Join form arguments for signature "a(...) ="
    b = ", ".join(formdata.argument_names)
    c = ", ".join(formdata.coefficient_names)
    arguments = "; ".join((b, c))
    signature = "%s(%s) = " % (formname, arguments, )

    # Define form as sum of integrals
    lines = []
    a = signature
    p = ""
    for itg in integrals:
        # TODO: Get list of expression strings instead of single expression!
        integrand_string = expression2latex(itg.integrand(), formdata.argument_names, formdata.coefficient_names)

        domain_type = itg.domain_type()
        dstr = domain_strings[domain_type]

        domain_id = itg.domain_id()
        if isinstance(domain_id, int):
            dstr += "_{%d}" % domain_id
        elif isinstance(domain_id, Domain) or domain_id == Measure.DOMAIN_ID_EVERYWHERE:
            pass
        elif domain_id == Measure.DOMAIN_ID_OTHERWISE:
            dstr += "_{\text{oth}}"
        elif isinstance(domain_id, tuple):
            dstr += "_{%s}" % domain_id

        b = p + "\\int_{%s}" % (dstr,)
        dxstr = Measure._domain_types[domain_type]
        dxstr = dx_strings[domain_type]
        c = "{ %s } \\,%s" % (integrand_string, dxstr)
        lines.append((a, b, c))
        a = "{}"
        p = "{}+ "

    sections.append(("Form", align(lines)))

    return sections

def ufl2latex(expression):
    "Generate LaTeX code for a UFL expression or form (wrapper for form2latex and expression2latex)."
    if isinstance(expression, Form):
        form_data = expression.compute_form_data()
        preprocessed_form = form_data.preprocessed_form
        return form2latex(preprocessed_form, form_data)
    else:
        return expression2latex(expression)

# --- LaTeX rendering of composite UFL objects ---

def deps2latex(deps):
    return "Dependencies: ${ %s }$." % ", ".join(sorted(deps))

def dependency_sorting(deplist, rank):
    #print "deplist = ", deplist

    def split(deps, state):
        left = []
        todo = []
        for dep in deps:
            if dep - state:
                left.append(dep)
            else:
                todo.append(dep)
        return todo, left

    deplistlist = []
    state = set()
    left = deplist

    # --- Initialization time
    #state.remove("x")
    precompute, left = split(left, state)
    deplistlist.append(precompute)

    state.add("x")
    precompute_quad, left = split(left, state)
    deplistlist.append(precompute_quad)

    # Permutations of 0/1 dependence of arguments
    indices = compute_indices((2,)*rank)
    for bfs in indices[1:]: # skip (0,...,0), already handled that
        for i, bf in reversed(list(enumerate(bfs))):
            n = "v%d" % i
            if bf:
                if n in state:
                    state.remove(n)
            else:
                state.add(n)
        next, left = split(left, state)
        deplistlist.append(next)

    # --- Runtime
    state.add("c")
    state.add("w")

    state.remove("x")
    runtime, left = split(left, state)
    deplistlist.append(runtime)

    state.add("x")
    runtime_quad, left = split(left, state)
    deplistlist.append(runtime_quad)

    indices = compute_indices((2,)*rank)
    for bfs in indices[1:]: # skip (0,...,0), already handled that
        for i, bf in reversed(list(enumerate(bfs))):
            n = "v%d" % i
            if bf:
                state.add(n)
            else:
                if n in state:
                    state.remove(n)
        next, left = split(left, state)
        deplistlist.append(next)

    ufl_assert(not left, "Shouldn't have anything left!")

    #print
    #print "Created deplistlist:"
    #for deps in deplistlist:
    #    print
    #    print "--- New stage:"
    #    print "\n".join(map(str, deps))
    #print

    return deplistlist

def code2latex(G, partitions, formdata):
    "TODO: Document me"
    bfn = formdata.argument_names
    cfn = formdata.coefficient_names

    V, E = G
    Vout = extract_outgoing_vertex_connections(G)

    # Sort dependency sets in a sensible way (preclude to a good quadrature code generator)
    deplistlist = dependency_sorting(partitions.keys(), len(bfn))

    def format_v(i):
        return "s_{%d}" % i

    pieces = []
    for deplist in deplistlist:
        #pieces.append("\n\n(Debugging: getting next list of dependencies)")
        for dep in deplist:
            lines = []
            for iv in partitions[dep]:
                v = V[iv]
                vout = Vout[iv]
                vl = format_v(iv)
                args = ", ".join(format_v(i) for i in vout)
                if args:
                    el = r"{\mbox{%s}}(%s)" % (v._uflclass.__name__, args)
                else: # terminal
                    el = r"{\mbox{%s}}" % (repr(v),)
                lines.append((vl, "= " + el))
            pieces.extend(("\n", deps2latex(dep), align(lines)))

    # Add final variable representing integrand
    vl = format_v(len(V)-1)
    pieces.append("\n")
    pieces.append("Variable representing integrand: %s" % vl)

    # Could also return list of (title, body) parts for subsections if wanted
    body = "\n".join(pieces)
    return body

def integrand2code(integrand, formdata):
    G = build_graph(integrand)
    partitions, keys = partition(G)
    return G, partitions

def formdata2latex(formdata): # TODO: Format better
    return verbatim(str(formdata))

def form2code2latex(formdata):
    # Render introductory sections
    title = "Form data"
    body = formdata2latex(formdata)
    sections = [(title, body)]

    # Render each integral as a separate section
    for itg in formdata.form.integrals(Measure.CELL):
        title = "%s integral over domain %d" % (itg.domain_type(), itg.domain_id())

        G, partitions = integrand2code(itg.integrand(), formdata)

        body = code2latex(G, partitions, formdata)

        sections.append((title, body))

    return sections

# --- Creating complete documents ---

def forms2latexdocument(forms, uflfilename, compile=False):
    "Render forms from a .ufl file as a LaTeX document."

    # Render one section for each form
    sections = []
    for form in forms:

        # Compute form data
        form = preprocess(form)
        form_data = form.form_data()

        # Generate LaTex code
        title = "Form %s" % form_data.name
        if compile:
            body = form2code2latex(form, form_data)
        else:
            body = form2latex(form, form_data)
        sections.append((title, body))

    # Render title
    suffix = "from UFL file %s" % uflfilename.replace("_", "\\_")
    if compile:
        title = "Compiled forms " + suffix
    else:
        title = "Forms " + suffix
    return document(title, sections)

# --- File operations ---

def ufl2tex(uflfilename, latexfilename, compile=False):
    "Compile a .tex file from a .ufl file."
    forms = load_forms(uflfilename)
    latex = forms2latexdocument(forms, uflfilename, compile)
    write_file(latexfilename, latex)

def tex2pdf(latexfilename, pdffilename):
    "Compile a .pdf file from a .tex file."
    pdflatex(latexfilename, pdffilename)
    openpdf(pdffilename)

def ufl2pdf(uflfilename, latexfilename, pdffilename, compile=False):
    "Compile a .pdf file from a .ufl file."
    ufl2tex(uflfilename, latexfilename, compile)
    tex2pdf(latexfilename, pdffilename)
