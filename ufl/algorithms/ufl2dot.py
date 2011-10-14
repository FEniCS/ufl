"""A collection of utility algorithms for printing
of UFL objects in the DOT graph visualization language,
mostly intended for debugging purposers."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# Last changed: 2009-06-19

from itertools import chain

from ufl.log import error
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.form import Form
from ufl.variable import Variable
from ufl.constantvalue import ScalarValue
from ufl.geometry import FacetNormal

# TODO: Maybe this can be cleaner written using the graph utilities

class2label = { \
    "IndexSum": "&sum;",
    "Sum": "&sum;",
    "Product": "&prod;",
    "Division": "/",
    "Inner": ":",
    "Dot": "&sdot;",
    "Outer": "&otimes;",
    "Grad": "grad",
    "Div": "div",
    "NablaGrad": "&nabla;&otimes;",
    "NablaDiv": "&nabla;&sdot;",
    "Curl": "&nabla;&times;",
    }

def build_entities(e, nodes, edges, nodeoffset):
    # Cutoff if we have handled e before
    if id(e) in nodes:
        return
    
    # Special-case Variable instances
    if isinstance(e, Variable):
        ops = (e._expression,)
        label = "var %d" % e._label._count
    else:
        ops = e.operands()
        if isinstance(e, Terminal):
            if isinstance(e, ScalarValue):
                label = repr(e._value)
            elif isinstance(e, FacetNormal):
                label = "n"
            else:
                label = repr(e)
        else:
            label = e._uflclass.__name__.split(".")[-1]
            if label in class2label:
                label = class2label[label]
    
    # Create node for parent e
    nodename = "n%04d" % (len(nodes) + nodeoffset)
    nodes[id(e)] = (nodename, label)
    
    # Handle all children recursively
    n = len(ops)
    oplabels = [None]*n
    if n == 2:
        oplabels = ["left", "right"]
    elif n > 2:
        oplabels = ["op%d" % i for i in range(n)]
    for i, o in enumerate(ops):
        # Handle entire subtree for expression o
        build_entities(o, nodes, edges, nodeoffset)
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

integralgraphformat = """  %s [label="Integral %s"]
  form_%s -> %s ;
  %s -> %s ;
%s"""

exprgraphformat = """  digraph ufl_expression
  {
  %s
  }"""

def ufl2dot(expression, formname="a", nodeoffset=0, begin=True, end=True):
    if isinstance(expression, Form):
        form = expression
        ci = form.cell_integrals()
        ei = form.exterior_facet_integrals()
        ii = form.interior_facet_integrals()
        mi = form.macro_cell_integrals()
        
        subgraphs = []
        nodes = {}
        edges = []
        for itg in chain(ci, ei, ii, mi):
            integrallabel = "%s%s" % (itg.measure().domain_type(), itg.measure().domain_id())
            integrand = itg.integrand()
            build_entities(integrand, nodes, edges, nodeoffset)
            rootnode = nodes[id(integrand)][0]
            entitylist = format_entities(nodes, edges)
            integralnode = "%s_%s" % (formname, integrallabel)
            subgraphs.append(integralgraphformat % (integralnode, integrallabel, formname, integralnode, integralnode, rootnode, entitylist))
        
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
        build_entities(integrand, nodes, edges, nodeoffset)
        entitylist = format_entities(nodes, edges)
        s = exprgraphformat % entitylist
    
    else:
        error("Invalid object type %s" % type(expression))
    
    return s, len(nodes) + nodeoffset

