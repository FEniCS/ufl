#!/usr/bin/env python

"""
Utility algorithms for inspection, conversion or transformation
of UFL objects in various ways.

(Currently, some utility functions are located in visitor.py,
some in traversal.py, and some in transformers.py,
depending on the method of implementation.
This file should contain userfriendly front-ends
to all the utility algorithms that we want to expose.)
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-04-01"

from base import *
from visitor import *
from transformers import *
from traversal import *

### Utilities to extract information from an expression:

# TODO: test performance of visitor implementations vs functional implementations

def basisfunctions(a):
    "Build a sorted list of all BasisFunctions in Form, Integral or expression."
    # FIXME: handle Form or Integral
    from visitor import BasisFunctionFinder
    vis = BasisFunctionFinder()
    vis.visit(u)
    # FIXME: sort by index
    return vis.basisfunctions

def functions(a):
    "Build a sorted list of all Functions in Form, Integral or expression."
    # FIXME: handle Form or Integral
    from visitor import CoefficientFinder
    vis = CoefficientFinder()
    vis.visit(u)
    # FIXME: sort by index
    return vis.coefficients

def duplications(u):
    from visitor import SubtreeFinder
    vis = SubtreeFinder()
    vis.visit(f)
    return vis.duplicated


### Utilities to convert expression to a different form:

def flatten(u):
    "Flatten (a+b)+(c+d) into a (a+b+c+d) and (a*b)*(c*d) into (a*b*c*d)."
    vis = TreeFlattener()
    return vis.visit(f)

def apply_summation(u):
    "Expand all repeated indices into explicit sums with fixed indices."
    ufl_error("Not implemented")
    # FIXME: Implement

def discover_indices(u):
    "Convert explicit sums into implicit sums (repeated indices)."
    ufl_error("Not implemented")
    # FIXME: Implement (like FFCs simplify done by Marie)

