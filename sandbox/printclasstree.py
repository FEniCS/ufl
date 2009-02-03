#!/usr/bin/env python

from ufl.classes import all_ufl_classes

vertices = []
for subclass in all_ufl_classes:
    vertices.append(subclass.__name__)

edges = []
for subclass in all_ufl_classes:
    superclass = subclass.mro()[1]
    edges.append((subclass.__name__, superclass.__name__))

dot = """digraph {
%s
%s
}""" % ("\n".join("  %s;" % v for v in vertices), "\n".join("  %s -> %s;" % e for e in edges))
print dot

