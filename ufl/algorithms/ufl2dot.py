"""A collection of utility algorithms for printing
of UFL objects in the DOT graph visualization language,
mostly intended for debugging purposers."""

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
# First added:  2008-11-17
# Last changed: 2011-10-14

from itertools import chain

from ufl.log import error
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.form import Form
from ufl.variable import Variable
from ufl.constantvalue import ScalarValue
from ufl.geometry import FacetNormal
from ufl.classes import FormArgument, MultiIndex, NegativeRestricted, PositiveRestricted
from ufl.algorithms.multifunction import MultiFunction

class ReprLabeller(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
    def terminal(self, e):
        return repr(e)
    def operator(self, e):
        return e._uflclass.__name__.split(".")[-1]

class CompactLabeller(ReprLabeller):
    def __init__(self, function_mapping=None):
        ReprLabeller.__init__(self)
        self._function_mapping = function_mapping

    # Terminals:
    def scalar_value(self, e):
        return repr(e._value)
    def zero(self, e):
        return "0"
    def identity(self, e):
        return "Id"
    def multi_index(self, e):
        return str(e)

    def form_argument(self, e):
        return self._function_mapping.get(id(e)) or str(e)

    def spatial_coordinate(self, e):
        return "x"
    def facet_normal(self, e):
        return "n"
    def circumradius(self, e):
        return "circumradius"
    def cell_volume(self, e):
        return "cell volume"
    def cell_surface_area(self, e):
        return "surface area"
    def max_facet_edge_length(self, e):
        return "max facet edge length"
    def facet_area(self, e):
        return "facet area"
    def facet_diameter(self, e):
        return "facet diameter"

    # Operators:
    def sum(self, e):
        return "+"
    def product(self, e):
        return "*"
    def division(self, e):
        return "/"
    def power(self, e):
        return "**"
    def math_function(self, e):
        return e._name
    def index_sum(self, e):
        return "&sum;"
    def indexed(self, e):
        return "[]"
    def component_tensor(self, e): # TODO: Understandable short notation for this?
        return "]["
    def negative_restricted(self, e):
        return "[-]"
    def positive_restricted(self, e):
        return "[+]"
    def cell_avg(self, e): # TODO: Understandable short notation for this?
        return "_K_"
    def facet_avg(self, e): # TODO: Understandable short notation for this?
        return "_F_"

    def inner(self, e):
        return "inner"
    def dot(self, e):
        return "dot"
    def outer(self, e):
        return "outer"
    def transposed(self, e):
        return "transp."
    def determinant(self, e):
        return "det"
    def trace(self, e):
        return "tr"
    def dev(self, e):
        return "dev"
    def skew(self, e):
        return "skew"

    def grad(self, e):
        return "grad"
    def div(self, e):
        return "div"
    def curl(self, e):
        return "curl"
    def nabla_grad(self, e):
        return "nabla_grad"
    def nabla_div(self, e):
        return "nabla_div"

    def diff(self, e):
        return "diff"

# Make this class like the ones above to use fancy math symbol labels
class2label = { \
    "IndexSum":  "&sum;",
    "Sum":       "&sum;",
    "Product":   "&prod;",
    "Division":  "/",
    "Inner":     ":",
    "Dot":       "&sdot;",
    "Outer":     "&otimes;",
    "Grad":      "grad",
    "Div":       "div",
    "NablaGrad": "&nabla;&otimes;",
    "NablaDiv":  "&nabla;&sdot;",
    "Curl":      "&nabla;&times;",
    }
class FancyLabeller(CompactLabeller):
    pass

def build_entities(e, nodes, edges, nodeoffset, prefix="", labeller=None):
    # TODO: Maybe this can be cleaner written using the graph utilities.
    # TODO: To collapse equal nodes with different objects, do not use id as key. Make this an option?

    # Cutoff if we have handled e before
    if id(e) in nodes:
        return
    if labeller is None:
        labeller = ReprLabeller()

    # Special-case Variable instances
    if isinstance(e, Variable): # FIXME: Is this really necessary?
        ops = (e._expression,)
        label = "variable %d" % e._label._count
    else:
        ops = e.operands()
        label = labeller(e)

    # Create node for parent e
    nodename = "%sn%04d" % (prefix, len(nodes) + nodeoffset)
    nodes[id(e)] = (nodename, label)

    # Handle all children recursively
    n = len(ops)
    if n == 2:
        #oplabels = ["left", "right"]
        oplabels = ["L", "R"]
    elif n > 2:
        oplabels = ["op%d" % i for i in range(n)]
    else:
        oplabels = [None]*n

    for i, o in enumerate(ops):
        # Handle entire subtree for expression o
        build_entities(o, nodes, edges, nodeoffset, prefix, labeller)

        # Add edge between e and child node o
        edges.append((id(e), id(o), oplabels[i]))

def format_entities(nodes, edges):
    entities = []
    for (nodename, label) in nodes.itervalues():
        node = '  %s [label="%s"];' % (nodename, label)
        entities.append(node)
    for (aid, bid, label) in edges:
        anodename = nodes[aid][0]
        bnodename = nodes[bid][0]
        if label is None:
            edge = '  %s -> %s ;' % (anodename, bnodename)
        else:
            edge = '  %s -> %s [label="%s"] ;' % (anodename, bnodename, label)
        entities.append(edge)
    return "\n".join(entities)

integralgraphformat = """  %(node)s [label="%(label)s"]
  form_%(formname)s -> %(node)s ;
  %(node)s -> %(root)s ;
%(entities)s"""

exprgraphformat = """  digraph ufl_expression
  {
  %s
  }"""

def ufl2dot(expression, formname="a", nodeoffset=0, begin=True, end=True, labeling="repr", object_names=None):
    if labeling == "repr":
        labeller = ReprLabeller()
    elif labeling == "compact":
        labeller = CompactLabeller(object_names or {})
        print object_names

    if isinstance(expression, Form):
        form = expression
        ci = form.integrals(Measure.CELL)
        ei = form.integrals(Measure.EXTERIOR_FACET)
        ii = form.integrals(Measure.INTERIOR_FACET)
        pi = form.integrals(Measure.POINT)
        mi = form.integrals(Measure.MACRO_CELL)

        subgraphs = []
        k = 0
        for itgs in (ci, ei, ii, pi, mi):
            for itg in itgs:
                prefix = "itg%d_" % k
                integralkey = "%s%s" % (itg.measure().domain_type(), itg.measure().domain_id())

                integrallabel = "%s %s" % (itg.measure().domain_type().capitalize().replace("_", " "), "integral")
                if len(itgs) > 1:
                    integrallabel += " %s" % (itg.measure().domain_id(),)

                integrand = itg.integrand()

                nodes = {}
                edges = []

                build_entities(integrand, nodes, edges, nodeoffset, prefix, labeller)
                rootnode = nodes[id(integrand)][0]
                entitylist = format_entities(nodes, edges)
                integralnode = "%s_%s" % (formname, integralkey)
                subgraphs.append(integralgraphformat % {
                    'node': integralnode,
                    'label': integrallabel,
                    'formname': formname,
                    'root': rootnode,
                    'entities': entitylist,
                    })
                nodeoffset += len(nodes)

        s = ""
        if begin:
            s += 'digraph ufl_form\n{\n  node [shape="box"] ;\n'
        s += '  form_%s [label="Form %s"] ;' % (formname, formname)
        s += "\n".join(subgraphs)
        if end:
            s += "\n}"

    elif isinstance(expression, Expr):
        nodes = {}
        edges = []

        build_entities(integrand, nodes, edges, nodeoffset, labeller)
        entitylist = format_entities(nodes, edges)
        s = exprgraphformat % entitylist

        nodeoffset += len(nodes)

    else:
        error("Invalid object type %s" % type(expression))

    return s, nodeoffset
