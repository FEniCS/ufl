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
# Modified by Paul T. Kühner, 2025

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
from ufl.argument import (
    Argument,
    Coargument,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
)
from ufl.averaging import CellAvg, FacetAvg
from ufl.cell import AbstractCell, Cell, TensorProductCell
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
from ufl.core.multiindex import FixedIndex, Index, IndexBase, MultiIndex
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
from ufl.domain import AbstractDomain, Mesh, MeshView
from ufl.equation import Equation
from ufl.exprcontainers import ExprList, ExprMapping
from ufl.finiteelement import AbstractFiniteElement
from ufl.form import BaseForm, Form, FormSum, ZeroBaseForm
from ufl.functionspace import (
    AbstractFunctionSpace,
    DualSpace,
    FunctionSpace,
    MixedFunctionSpace,
    TensorProductFunctionSpace,
)
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
from ufl.integral import Integral
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
from ufl.measure import Measure, MeasureProduct, MeasureSum
from ufl.pullback import (
    AbstractPullback,
    ContravariantPiola,
    CovariantContravariantPiola,
    CovariantPiola,
    CustomPullback,
    DoubleContravariantPiola,
    DoubleCovariantPiola,
    IdentityPullback,
    L2Piola,
    MixedPullback,
    NonStandardPullbackException,
    PhysicalPullback,
    SymmetricPullback,
    UndefinedPullback,
)
from ufl.referencevalue import ReferenceValue
from ufl.restriction import NegativeRestricted, PositiveRestricted, Restricted
from ufl.sobolevspace import DirectionalSobolevSpace, SobolevSpace
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

__all__ = [
    "EQ",
    "GE",
    "GT",
    "LE",
    "LT",
    "NE",
    "Abs",
    "AbstractCell",
    "AbstractDomain",
    "AbstractFiniteElement",
    "AbstractFunctionSpace",
    "AbstractPullback",
    "Acos",
    "Action",
    "Adjoint",
    "AndCondition",
    "Argument",
    "Asin",
    "Atan",
    "Atan2",
    "BaseForm",
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
    "Cell",
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
    "ContravariantPiola",
    "CoordinateDerivative",
    "Cos",
    "Cosh",
    "CovariantContravariantPiola",
    "CovariantPiola",
    "Cross",
    "Curl",
    "CustomPullback",
    "Derivative",
    "Determinant",
    "Deviatoric",
    "DirectionalSobolevSpace",
    "Div",
    "Division",
    "Dot",
    "DoubleContravariantPiola",
    "DoubleCovariantPiola",
    "DualSpace",
    "Equation",
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
    "FixedIndex",
    "FloatValue",
    "Form",
    "Form",
    "FormArgument",
    "FormSum",
    "FunctionSpace",
    "GeometricCellQuantity",
    "GeometricFacetQuantity",
    "GeometricQuantity",
    "GeometricRidgeQuantity",
    "Grad",
    "Identity",
    "IdentityPullback",
    "Imag",
    "Index",
    "IndexBase",
    "IndexSum",
    "Indexed",
    "Inner",
    "IntValue",
    "Integral",
    "Interpolate",
    "Inverse",
    "Jacobian",
    "JacobianDeterminant",
    "JacobianInverse",
    "L2Piola",
    "Label",
    "ListTensor",
    "Ln",
    "MathFunction",
    "Matrix",
    "MaxCellEdgeLength",
    "MaxFacetEdgeLength",
    "MaxValue",
    "Measure",
    "MeasureProduct",
    "MeasureSum",
    "Mesh",
    "MeshView",
    "MinCellEdgeLength",
    "MinFacetEdgeLength",
    "MinValue",
    "MixedFunctionSpace",
    "MixedPullback",
    "MultiIndex",
    "NablaDiv",
    "NablaGrad",
    "NegativeRestricted",
    "NonStandardPullbackException",
    "NotCondition",
    "Operator",
    "OrCondition",
    "Outer",
    "PermutationSymbol",
    "Perp",
    "PhysicalPullback",
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
    "SobolevSpace",
    "SpatialCoordinate",
    "Sqrt",
    "Sum",
    "Sym",
    "SymmetricPullback",
    "Tan",
    "Tanh",
    "TensorProductCell",
    "TensorProductFunctionSpace",
    "Terminal",
    "TestFunction",
    "TestFunctions",
    "Trace",
    "Transposed",
    "TrialFunction",
    "TrialFunctions",
    "UndefinedPullback",
    "Variable",
    "VariableDerivative",
    "Zero",
    "ZeroBaseForm",
    "ZeroBaseForm",
    "__exproperators",
    "abstract_classes",
    "all_ufl_classes",
    "nonterminal_classes",
    "terminal_classes",
    "ufl_classes",
]

# Collect all classes in sets automatically classified by some properties
all_ufl_classes = set(ufl.core.expr.Expr._ufl_all_classes_)
abstract_classes = set(c for c in all_ufl_classes if c._ufl_is_abstract_)
ufl_classes = set(c for c in all_ufl_classes if not c._ufl_is_abstract_)
terminal_classes = set(c for c in all_ufl_classes if c._ufl_is_terminal_)
nonterminal_classes = set(c for c in all_ufl_classes if not c._ufl_is_terminal_)
