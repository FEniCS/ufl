"""This module defines evaluation algorithms for converting
converting UFL expressions to swiginac representation."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2008-08-22"

from collections import defaultdict

from ..common import some_key, product
from ..output import ufl_assert, ufl_error

# All classes:
from ..base import UFLObject, Terminal, Number
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunctions import BasisFunction, Function, Constant
#from ..basisfunctions import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index
#from ..indexing import FixedIndex, AxisType, as_index, as_index_tuple, extract_indices
from ..tensors import ListVector, ListMatrix, Tensor
#from ..tensors import Vector, Matrix
from ..algebra import Sum, Product, Division, Power, Mod, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral
#from ..formoperators import Derivative, Action, Rhs, Lhs # TODO: What to do with these?

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients, indices

import swiginac


class Context:
    "Context class for obtaining terminal expressions."
    def basisfunction(self, i):
        return NotImplemented
    
    def coefficient(self, i):
        return NotImplemented
    
    def facet_normal(self):
        return NotImplemented
    
    def variable(self, i):
        return None


# FIXME: Implement for all UFL expressions!
# Steps:
# - Handle scalar expressions only
# - Handle indexed tensor expressions (i.e. after expand_compounds)
# - Handle variable lists (cannot express with swiginac, need to represent code structure)
# - Handle derivatives d/ds and d/dx with AD (need this for quadrature code)
# - Handle conditionals (cannot express with swiginac, need to represent code structure)


def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    return handlers[expression.__class__](expression, *ops)


def swiginac_handlers(context):
    
    sw = swiginac

    # Show a clear error message if we miss some types here:
    def not_implemented(x, ops):
        ufl_error("No handler defined for %s in swiginac_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    
    ### Basic terminal objects:
    def s_number(x):
        return sw.numeric(x._value)
    d[Number] = s_number
    
    def s_variable(x):
        # Lookup variable and evaluate its expression directly if not found
        v = context.variable(x._count)
        if v is None:
            return evaluate_as_swiginac(x._expression, context)
        return v
    d[Variable] = s_variable

    def s_basisfunction(x):
        return context.basisfunction(x._count)
    d[BasisFunction] = s_basisfunction

    def s_function(x):
        return context.function(x._count)
    d[Function] = s_function
    d[Constant] = s_function

    def s_facet_normal(x):
        return context.facet_normal()
    d[FacetNormal] = s_facet_normal
    
    ### Basic algebra:
    def s_sum(x, *ops):
        return sum(ops)
    d[Sum] = s_sum

    def s_product(x, *ops):
        return product(ops)
    d[Product] = s_product

    def s_division(x, a, b):
        return a / b
    d[Division] = s_division

    def s_power(x, a, b):
        return a ** b
    d[Power] = s_power

    #def s_mod(x, a, b):
    #    return a % b
    #d[Mod] = s_mod

    def s_abs(x, a):
        return abs(a)
    d[Abs] = s_abs

    ### Basic math functions:
    def s_sqrt(x, y):
        return sw.sqrt(y)
    d[Sqrt] = s_sqrt
    
    def s_exp(x, y):
        return sw.exp(y)
    d[Exp] = s_exp
    
    def s_ln(x, y):
        return sw.ln(y)
    d[Ln] = s_ln
    
    def s_cos(x, y):
        return sw.cos(y)
    d[Cos] = s_cos
    
    def s_sin(x, y):
        return sw.sin(y)
    d[Sin] = s_sin

    ### Index handling:
    #d[Indexed] = s_Indexed # TODO
    #d[MultiIndex] = s_MultiIndex # TODO

    ### Container handling:
    #d[ListVector] = s_ListVector # TODO
    #d[ListMatrix] = s_ListMatrix # TODO
    #d[Tensor] = s_Tensor # TODO
    
    ### Differentiation:
    #d[PartialDerivative] = s_PartialDerivative # TODO
    #d[Diff] = s_Diff # TODO

    ### Interior facet stuff:
    #d[Restricted] = s_Restricted # TODO
    #d[PositiveRestricted] = s_PositiveRestricted # TODO
    #d[NegativeRestricted] = s_NegativeRestricted # TODO
    
    ### Requires code structure:
    #d[EQ] = s_EQ # TODO
    #d[NE] = s_NE # TODO
    #d[LE] = s_LE # TODO
    #d[GE] = s_GE # TODO
    #d[LT] = s_LT # TODO
    #d[GT] = s_GT # TODO
    #d[Conditional] = s_Conditional # TODO

    ### Replaced by expand_compounds:
    #d[Identity] = s_Identity # TODO
    #d[Transposed] = s_Transposed # TODO
    #d[Outer] = s_Outer # TODO
    #d[Inner] = s_Inner # TODO
    #d[Dot] = s_Dot # TODO
    #d[Cross] = s_Cross # TODO
    #d[Trace] = s_Trace # TODO
    #d[Determinant] = s_Determinant # TODO
    #d[Inverse] = s_Inverse # TODO
    #d[Deviatoric] = s_Deviatoric # TODO
    #d[Cofactor] = s_Cofactor # TODO
    
    ### Replaced by expand_compounds:
    #d[Grad] = s_Grad # TODO
    #d[Div] = s_Div # TODO
    #d[Curl] = s_Curl # TODO
    #d[Rot] = s_Rot # TODO

    return d


def evaluate_as_swiginac(expression, context):
    d = swiginac_handlers(context)
    return transform(expression, d)
