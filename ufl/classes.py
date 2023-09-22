"""Classes.

This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import CellFacetooBar" for getting
implementation details not exposed through the default ufl namespace.
It also contains functionality used by algorithms for dealing with groups
of classes, and for mapping types to different handler functions.
"""
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

import ufl.algebra
import ufl.argument
import ufl.averaging
import ufl.cell
import ufl.coefficient
import ufl.conditional
import ufl.constantvalue
import ufl.core.expr
import ufl.core.multiindex
import ufl.core.operator
import ufl.core.terminal
import ufl.differentiation
import ufl.domain
import ufl.equation
import ufl.exprcontainers
import ufl.finiteelement
import ufl.form
import ufl.functionspace
import ufl.geometry
import ufl.indexed
import ufl.indexsum
import ufl.integral
import ufl.mathfunctions
import ufl.measure
import ufl.pull_back
import ufl.referencevalue
import ufl.restriction
import ufl.sobolevspace
import ufl.tensoralgebra
import ufl.tensors
import ufl.variable
from ufl import exproperators as __exproperators

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
    "__exproperators",
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


__all__ += populate_namespace_with_module_classes(ufl.cell, locals())
__all__ += populate_namespace_with_module_classes(ufl.finiteelement, locals())
__all__ += populate_namespace_with_module_classes(ufl.domain, locals())
__all__ += populate_namespace_with_module_classes(ufl.functionspace, locals())
__all__ += populate_namespace_with_module_classes(ufl.core.multiindex, locals())
__all__ += populate_namespace_with_module_classes(ufl.argument, locals())
__all__ += populate_namespace_with_module_classes(ufl.measure, locals())
__all__ += populate_namespace_with_module_classes(ufl.integral, locals())
__all__ += populate_namespace_with_module_classes(ufl.form, locals())
__all__ += populate_namespace_with_module_classes(ufl.equation, locals())
__all__ += populate_namespace_with_module_classes(ufl.pull_back, locals())
__all__ += populate_namespace_with_module_classes(ufl.sobolevspace, locals())
