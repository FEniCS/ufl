# -*- coding: utf-8 -*-
# flake8: noqa
"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl_legacy.classes import CellFacetooBar" for getting
implementation details not exposed through the default ufl_legacy namespace.
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

# This will be populated part by part below
__all__ = []


# Import all submodules, triggering execution of the ufl_type class
# decorator for each Expr class.

# Base classes of Expr type hierarchy
import ufl_legacy.core.expr
import ufl_legacy.core.terminal
import ufl_legacy.core.operator

# Terminal types
import ufl_legacy.constantvalue
import ufl_legacy.argument
import ufl_legacy.coefficient
import ufl_legacy.geometry

# Operator types
import ufl_legacy.averaging
import ufl_legacy.indexed
import ufl_legacy.indexsum
import ufl_legacy.variable
import ufl_legacy.tensors
import ufl_legacy.algebra
import ufl_legacy.tensoralgebra
import ufl_legacy.mathfunctions
import ufl_legacy.differentiation
import ufl_legacy.conditional
import ufl_legacy.restriction
import ufl_legacy.exprcontainers
import ufl_legacy.referencevalue

# Make sure we import exproperators which attaches special functions
# to Expr
from ufl_legacy import exproperators as __exproperators

# Make sure to import modules with new Expr subclasses here!

# Collect all classes in sets automatically classified by some properties
all_ufl_classes = set(ufl_legacy.core.expr.Expr._ufl_all_classes_)
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
    classes = ufl_legacy.core.expr.Expr._ufl_all_classes_
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


import ufl_legacy.cell  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.cell, locals())

import ufl_legacy.finiteelement  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.finiteelement, locals())

import ufl_legacy.domain  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.domain, locals())

import ufl_legacy.functionspace  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.functionspace, locals())

import ufl_legacy.core.multiindex  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.core.multiindex, locals())

import ufl_legacy.argument  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.argument, locals())

import ufl_legacy.measure  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.measure, locals())

import ufl_legacy.integral  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.integral, locals())

import ufl_legacy.form  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.form, locals())

import ufl_legacy.equation  # noqa E401
__all__ += populate_namespace_with_module_classes(ufl_legacy.equation, locals())
