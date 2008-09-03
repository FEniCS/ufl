"""Functions to check the validity of forms."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-18"

from ..output import UFLException, ufl_error, ufl_assert, ufl_info
from ..base import is_python_scalar, is_scalar, is_true_scalar

# ALl classes:
from ..base import UFLObject, Terminal, Scalar
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunctions import BasisFunction, Function, Constant
#from ..basisfunctions import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed
#from ..indexing import Index, FixedIndex, AxisType, as_index, as_index_tuple, extract_indices
from ..tensors import ListVector, ListMatrix, Tensor
#from ..tensors import Vector, Matrix
from ..algebra import Sum, Product, Division, Power, Mod, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral
from ..formoperators import Derivative, Action, Rhs, Lhs, rhs, lhs

from ..finiteelement import _domain2dim

# Other algorithms
from .traversal import post_traversal, post_walk, iter_expressions
from .analysis import value_shape, domain, elements
from .predicates import is_multilinear


def validate_form(a):
    """Performs all implemented validations on a form. Raises exception if something fails."""
    
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    
    ufl_assert(is_multilinear(a), "Form is not multilinear in the BasisFunction arguments.")
    
    dom = domain(a)
    for e in elements(a):
        ufl_assert(dom == e.domain(), "Inconsistent domains in form, got both %s and %s." % (dom, e.domain()))
    
    dim = _domain2dim[dom]
    for e in iter_expressions(a):
        ufl_assert(value_shape(e, dim) == (), "Got non-scalar integrand expression:\n%s" % e)

