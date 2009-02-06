#!/usr/bin/env python

from collections import defaultdict
from ufl.classes import all_ufl_classes

# Build lists of subclasses
subgraphs = defaultdict(list)
for c in all_ufl_classes:
    subgraphs[c.mro()[1].__name__].append(c.__name__)

# Recursive graph formatting
def format_children(parent, level, skipparent=False):
    children = subgraphs[parent]
    t = "  "*level
    begin = t + "subgraph {\n"
    end   = t + "}\n"
    g = ""
    for child in children:
        if child in subgraphs:
            g += begin
            g += format_children(child, level+1)
            g += end
        if not skipparent:
            g += t + "%s -> %s;\n" % (child, parent)
    return g

# Render graph body!
body = format_children("object", 1, True)

# Set global formatting options
format = """
  node [shape=box, style=filled, color=lightgrey];
  splines=true;
"""

# Combine everythig to a global graph
dot = """strict digraph {
%s
%s
}""" % (format, body)
print dot
