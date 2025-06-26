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
import ufl.pullback
import ufl.referencevalue
import ufl.restriction
import ufl.sobolevspace
import ufl.tensoralgebra
import ufl.tensors
import ufl.variable
from ufl import exproperators as __exproperators
from ufl.action import Action
from ufl.adjoint import Adjoint
from ufl.algebra import Abs, Conj, Division, Imag, Power, Product, Real, Sum
from ufl.argument import Argument, Coargument
from ufl.averaging import CellAvg, FacetAvg
from ufl.coefficient import Coefficient, Cofunction
from ufl.conditional import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    AndCondition,
    BinaryCondition,
    Condition,
    Conditional,
    MaxValue,
    MinValue,
    NotCondition,
    OrCondition,
)
from ufl.constant import Constant
from ufl.constantvalue import (
    ComplexValue,
    ConstantValue,
    FloatValue,
    Identity,
    IntValue,
    PermutationSymbol,
    RealValue,
    ScalarValue,
    Zero,
)
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.expr import Expr
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.core.multiindex import MultiIndex
from ufl.core.operator import Operator
from ufl.core.terminal import FormArgument, Terminal
from ufl.differentiation import (
    BaseFormCoordinateDerivative,
    BaseFormDerivative,
    BaseFormOperatorCoordinateDerivative,
    BaseFormOperatorDerivative,
    CoefficientDerivative,
    CompoundDerivative,
    CoordinateDerivative,
    Curl,
    Derivative,
    Div,
    Grad,
    NablaDiv,
    NablaGrad,
    ReferenceCurl,
    ReferenceDiv,
    ReferenceGrad,
    VariableDerivative,
)
from ufl.exprcontainers import ExprList, ExprMapping
from ufl.form import BaseForm, Form, FormSum, ZeroBaseForm
from ufl.geometry import (
    CellCoordinate,
    CellDiameter,
    CellEdgeVectors,
    CellFacetJacobian,
    CellFacetJacobianDeterminant,
    CellFacetJacobianInverse,
    CellFacetOrigin,
    CellNormal,
    CellOrientation,
    CellOrigin,
    CellRidgeJacobian,
    CellRidgeJacobianDeterminant,
    CellRidgeJacobianInverse,
    CellRidgeOrigin,
    CellVertices,
    CellVolume,
    Circumradius,
    FacetArea,
    FacetCoordinate,
    FacetEdgeVectors,
    FacetJacobian,
    FacetJacobianDeterminant,
    FacetJacobianInverse,
    FacetNormal,
    FacetOrientation,
    FacetOrigin,
    FacetRidgeJacobian,
    GeometricCellQuantity,
    GeometricFacetQuantity,
    GeometricQuantity,
    GeometricRidgeQuantity,
    Jacobian,
    JacobianDeterminant,
    JacobianInverse,
    MaxCellEdgeLength,
    MaxFacetEdgeLength,
    MinCellEdgeLength,
    MinFacetEdgeLength,
    QuadratureWeight,
    ReferenceCellEdgeVectors,
    ReferenceCellVolume,
    ReferenceFacetEdgeVectors,
    ReferenceFacetVolume,
    ReferenceNormal,
    ReferenceRidgeVolume,
    RidgeCoordinate,
    RidgeJacobian,
    RidgeJacobianDeterminant,
    RidgeJacobianInverse,
    RidgeOrigin,
    SpatialCoordinate,
)
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.mathfunctions import (
    Acos,
    Asin,
    Atan,
    Atan2,
    BesselFunction,
    BesselI,
    BesselJ,
    BesselK,
    BesselY,
    Cos,
    Cosh,
    Erf,
    Exp,
    Ln,
    MathFunction,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
)
from ufl.matrix import Matrix
from ufl.referencevalue import ReferenceValue
from ufl.restriction import NegativeRestricted, PositiveRestricted, Restricted
from ufl.tensoralgebra import (
    Cofactor,
    CompoundTensorOperator,
    Cross,
    Determinant,
    Deviatoric,
    Dot,
    Inner,
    Inverse,
    Outer,
    Perp,
    Skew,
    Sym,
    Trace,
    Transposed,
)
from ufl.tensors import ComponentTensor, ListTensor
from ufl.variable import Label, Variable

# Collect all classes in sets automatically classified by some properties
all_ufl_classes = set(ufl.core.expr.Expr._ufl_all_classes_)
abstract_classes = set(c for c in all_ufl_classes if c._ufl_is_abstract_)
ufl_classes = set(c for c in all_ufl_classes if not c._ufl_is_abstract_)
terminal_classes = set(c for c in all_ufl_classes if c._ufl_is_terminal_)
nonterminal_classes = set(c for c in all_ufl_classes if not c._ufl_is_terminal_)

__all__ += [
    "EQ",
    "GE",
    "GT",
    "LE",
    "LT",
    "NE",
    "Abs",
    "Acos",
    "Action",
    "Adjoint",
    "AndCondition",
    "Argument",
    "Asin",
    "Atan",
    "Atan2",
    "BaseForm",
    "BaseFormCoordinateDerivative",
    "BaseFormDerivative",
    "BaseFormOperator",
    "BaseFormOperatorCoordinateDerivative",
    "BaseFormOperatorDerivative",
    "BesselFunction",
    "BesselI",
    "BesselJ",
    "BesselK",
    "BesselY",
    "BinaryCondition",
    "CellAvg",
    "CellCoordinate",
    "CellDiameter",
    "CellEdgeVectors",
    "CellFacetJacobian",
    "CellFacetJacobianDeterminant",
    "CellFacetJacobianInverse",
    "CellFacetOrigin",
    "CellNormal",
    "CellOrientation",
    "CellOrigin",
    "CellRidgeJacobian",
    "CellRidgeJacobianDeterminant",
    "CellRidgeJacobianInverse",
    "CellRidgeOrigin",
    "CellVertices",
    "CellVolume",
    "Circumradius",
    "Coargument",
    "Coefficient",
    "CoefficientDerivative",
    "Cofactor",
    "Cofunction",
    "ComplexValue",
    "ComponentTensor",
    "CompoundDerivative",
    "CompoundTensorOperator",
    "Condition",
    "Conditional",
    "Conj",
    "Constant",
    "ConstantValue",
    "CoordinateDerivative",
    "Cos",
    "Cosh",
    "Cross",
    "Curl",
    "Derivative",
    "Determinant",
    "Deviatoric",
    "Div",
    "Division",
    "Dot",
    "Erf",
    "Exp",
    "Expr",
    "ExprList",
    "ExprMapping",
    "ExternalOperator",
    "FacetArea",
    "FacetAvg",
    "FacetCoordinate",
    "FacetEdgeVectors",
    "FacetJacobian",
    "FacetJacobianDeterminant",
    "FacetJacobianInverse",
    "FacetNormal",
    "FacetOrientation",
    "FacetOrigin",
    "FacetRidgeJacobian",
    "FloatValue",
    "Form",
    "FormArgument",
    "FormSum",
    "GeometricCellQuantity",
    "GeometricFacetQuantity",
    "GeometricQuantity",
    "GeometricRidgeQuantity",
    "Grad",
    "Identity",
    "Imag",
    "IndexSum",
    "Indexed",
    "Inner",
    "IntValue",
    "Interpolate",
    "Inverse",
    "Jacobian",
    "JacobianDeterminant",
    "JacobianInverse",
    "Label",
    "ListTensor",
    "Ln",
    "MathFunction",
    "Matrix",
    "MaxCellEdgeLength",
    "MaxFacetEdgeLength",
    "MaxValue",
    "MinCellEdgeLength",
    "MinFacetEdgeLength",
    "MinValue",
    "MultiIndex",
    "NablaDiv",
    "NablaGrad",
    "NegativeRestricted",
    "NotCondition",
    "Operator",
    "OrCondition",
    "Outer",
    "PermutationSymbol",
    "Perp",
    "PositiveRestricted",
    "Power",
    "Product",
    "QuadratureWeight",
    "Real",
    "RealValue",
    "ReferenceCellEdgeVectors",
    "ReferenceCellVolume",
    "ReferenceCurl",
    "ReferenceDiv",
    "ReferenceFacetEdgeVectors",
    "ReferenceFacetVolume",
    "ReferenceGrad",
    "ReferenceNormal",
    "ReferenceRidgeVolume",
    "ReferenceValue",
    "Restricted",
    "RidgeCoordinate",
    "RidgeJacobian",
    "RidgeJacobianDeterminant",
    "RidgeJacobianInverse",
    "RidgeOrigin",
    "ScalarValue",
    "Sin",
    "Sinh",
    "Skew",
    "SpatialCoordinate",
    "Sqrt",
    "Sum",
    "Sym",
    "Tan",
    "Tanh",
    "Terminal",
    "Trace",
    "Transposed",
    "Variable",
    "VariableDerivative",
    "Zero",
    "ZeroBaseForm",
    "__exproperators",
    "abstract_classes",
    "all_ufl_classes",
    "nonterminal_classes",
    "terminal_classes",
    "ufl_classes",
]


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
__all__ += populate_namespace_with_module_classes(ufl.pullback, locals())
__all__ += populate_namespace_with_module_classes(ufl.sobolevspace, locals())
