# -*- coding: utf-8 -*-
"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import CellFacetooBar" for getting
implementation details not exposed through the default ufl namespace.
It also contains functionality used by algorithms for dealing with groups
of classes, and for mapping types to different handler functions."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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
# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2011
# Modified by Andrew T. T. McRae, 2014


# This will be populated part by part below
__all__ = []


#
# Import all submodules, triggering execution of the
# ufl_type class decorator for each Expr class.
#

# Base classes of Expr type hierarchy
import ufl.core.expr
import ufl.core.terminal
import ufl.core.operator

# Terminal types
import ufl.constantvalue
import ufl.argument
import ufl.coefficient
import ufl.geometry
import ufl.indexing

# Operator types
import ufl.indexed
import ufl.indexsum
import ufl.variable
import ufl.tensors
import ufl.algebra
import ufl.tensoralgebra
import ufl.mathfunctions
import ufl.differentiation
import ufl.conditional
import ufl.restriction
import ufl.exprcontainers
import ufl.referencevalue

# Make sure we import exproperators which attaches special functions to Expr
from ufl import exproperators as __exproperators


#
# Make sure to import modules with new Expr subclasses here!
#

# Collect all classes in sets automatically classified by some properties
all_ufl_classes     = set(ufl.core.expr.Expr._ufl_all_classes_)
abstract_classes    = set(c for c in all_ufl_classes if c._ufl_is_abstract_)
ufl_classes         = set(c for c in all_ufl_classes if not c._ufl_is_abstract_)
terminal_classes    = set(c for c in all_ufl_classes if c._ufl_is_terminal_)
nonterminal_classes = set(c for c in all_ufl_classes if not c._ufl_is_terminal_)

__all__ += [
    "all_ufl_classes",
    "abstract_classes",
    "ufl_classes",
    "terminal_classes",
    "nonterminal_classes",
    ]

def populate_namespace_with_expr_classes(namespace):
    """Export all Expr subclasses into the namespace under their natural name."""
    names = []
    classes = ufl.core.expr.Expr._ufl_all_classes_
    for cls in classes:
        class_name = cls.__name__
        namespace[class_name] = cls
        names.append(class_name)
    return names

__all__ += populate_namespace_with_expr_classes(locals())

# Domain types
from ufl.cell import Cell, ProductCell, OuterProductCell
from ufl.domain import Domain, ProductDomain

__all__ += [
    "Cell", "ProductCell", "OuterProductCell",
    "Domain", "ProductDomain",
    ]

# Elements
from ufl.finiteelement import (
    FiniteElementBase,
    FiniteElement,
    MixedElement, VectorElement, TensorElement,
    EnrichedElement, RestrictedElement,
    TensorProductElement, OuterProductElement, OuterProductVectorElement)

__all__ += [
    "FiniteElementBase",
    "FiniteElement",
    "MixedElement", "VectorElement", "TensorElement",
    "EnrichedElement", "RestrictedElement",
    "TensorProductElement", "OuterProductElement", "OuterProductVectorElement",
    ]

# Other non-Expr types
from ufl.argument import TestFunction, TrialFunction, TestFunctions, TrialFunctions
from ufl.core.multiindex import IndexBase, FixedIndex, Index

__all__ += [
    "TestFunction", "TrialFunction", "TestFunctions", "TrialFunctions",
    "IndexBase", "FixedIndex", "Index",
    ]

# Higher level abstractions
from ufl.measure import Measure, MeasureSum, MeasureProduct
from ufl.integral import Integral
from ufl.form import Form
from ufl.equation import Equation

__all__ += [
    "Measure", "MeasureSum", "MeasureProduct",
    "Integral",
    "Form",
    "Equation",
    ]
