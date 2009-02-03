#!/usr/bin/env python

from ufl.classes import all_ufl_classes

vertices = []
for subclass in all_ufl_classes:
    vertices.append(subclass.__name__)

edges = []
for subclass in all_ufl_classes:
    superclass = subclass.mro()[1]
    if not superclass is object:
        edges.append((subclass.__name__, superclass.__name__))

format1= ""
format2 = """
  node [shape=box];
"""
format3 = """
  root=Expr;
  maxiter=100000;
  splines=true;
  node [shape=box];
"""
format = format3

dot = """strict digraph {
%s
%s
%s
}""" % (format, "\n".join("  %s;" % v for v in vertices), "\n".join("  %s -> %s;" % e for e in edges))
print dot

