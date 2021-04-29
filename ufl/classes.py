# -*- coding: utf-8 -*-
# flake8: noqa
"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import CellFacetooBar" for getting
implementation details not exposed through the default ufl namespace.
It also contains functionality used by algorithms for dealing with groups
of classes, and for mapping types to different handler functions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2011
# Modified by Andrew T. T. McRae, 2014
# Modified by Nacime Bouziani, 2021

# This will be populated part by part below
__all__ = []


# Import all submodules, triggering execution of the ufl_type class
# decorator for each Expr class.

# Base classes of Expr type hierarchy
import ufl.core.expr
import ufl.core.terminal
import ufl.core.operator

# Terminal types
import ufl.constantvalue
import ufl.argument
import ufl.coefficient
import ufl.geometry

# Operator types
import ufl.averaging
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

# Make sure we import exproperators which attaches special functions
# to Expr
from ufl import exproperators as __exproperators

# Make sure to import modules with new Expr subclasses here!

# Collect all classes in sets automatically classified by some properties
all_ufl_classes = set(ufl.core.expr.Expr._ufl_all_classes_)
abstract_classes = set(c for c in all_ufl_classes if c._ufl_is_abstract_)
ufl_classes = set(c for c in all_ufl_classes if not c._ufl_is_abstract_)
terminal_classes = set(c for c in all_ufl_classes if c._ufl_is_terminal_)
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


# Semi-automated imports of non-expr classes:

def populate_namespace_with_module_classes(mod, loc):
    """Export the classes that submodules list in __all_classes__."""
    names = mod.__all_classes__
    for name in names:
        loc[name] = getattr(mod, name)
    return names


import ufl.cell  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.cell, locals())

import ufl.finiteelement  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.finiteelement, locals())

import ufl.domain  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.domain, locals())

import ufl.functionspace  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.functionspace, locals())

import ufl.core.multiindex  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.core.multiindex, locals())

import ufl.argument  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.argument, locals())

import ufl.measure  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.measure, locals())

import ufl.integral  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.integral, locals())

import ufl.form  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.form, locals())

import ufl.equation  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl.equation, locals())
