"""Functions to check the validity of forms."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-03"

# Modified by Anders Logg, 2008.

from ..output import UFLException, ufl_error, ufl_assert, ufl_info, ufl_warning
from ..base import is_python_scalar, is_scalar, is_true_scalar

# ALl classes:
from ..base import UFLObject, Terminal, FloatValue, ZeroType
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunction import BasisFunction
from ..function import Function
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
from ..differentiation import SpatialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral

from ..finiteelement import _domain2dim

# Other algorithms
from .traversal import post_traversal, post_walk, iter_expressions
from .analysis import value_shape, domain, elements
from .predicates import is_multilinear

def validate_form(form):
    """Performs all implemented validations on a form. Raises exception if something fails."""

    # Check that we get a form
    ufl_assert(isinstance(form, Form), "Expecting a Form.")

    # Check that form is multilinear
    is_ml = is_multilinear(form)
    if not is_ml:
        ufl_warning("Form is not multilinear according to buggy(!) is_multilinear function.")
    #ufl_assert(is_ml, "Form is not multilinear.")

    # Check that domain is the same for all elements
    dom = domain(form)
    for e in elements(form):
        ufl_assert(dom == e.domain(), "Inconsistent domains in form, got both %s and %s." % (dom, e.domain()))

    # Check that all integrands are scalar
    dim = _domain2dim[dom]
    for e in iter_expressions(form):
        ufl_assert(value_shape(e, dim) == (), "Got non-scalar integrand expression:\n%s" % e)
