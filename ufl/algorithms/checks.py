"""Functions to check the validity of forms."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-14"

from ..output import UFLException, ufl_error, ufl_assert, ufl_info
from ..base import is_python_scalar, is_scalar, is_true_scalar

# ALl classes:
from ..base import UFLObject, Terminal, Number
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
from ..tensoralgebra import Identity, Transpose, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import PartialDerivative, Diff, DifferentialOperator, Grad, Div, Curl, Rot
from ..form import Form
from ..integral import Integral
from ..formoperators import Derivative, Action, Rhs, Lhs, rhs, lhs

# Other algorithms
from .traversal import post_traversal, post_walk
#from .analysis import *
#from .predicates import *


def value_shape(expression, dim):
    """Evaluate the value shape of expression with given implicit dimension."""

    # TODO: Implement this for all non-compound stuff. Or do we need this?
    #       What we really want is perhaps to check the index dimensions, not the shapes...
    #       That can be done in the index remapping function.
    #elif isinstance(expression, Index):
    #    tmp = expression.dim()
    #    shape = (dim,) if tmp is None else (tmp,)

    if isinstance(expression, Terminal):
        if isinstance(expression, (BasisFunction, Function)):
            shape = expression._element.value_shape()
        elif isinstance(expression, (Number, Constant)):
            shape = ()
        elif isinstance(expression, FacetNormal):
            shape = (dim,)
        elif isinstance(expression, Identity):
            shape = (dim, dim)
        else:
            ufl_error("Missing %s handler for computing value shape." % expression.__class__)
    else:
        ops_shapes = [value_shape(o, dim) for o in expression.operands()]

        if isinstance(expression, Sum):
            shape = ops_shapes[0]
            ufl_assert(all(o == shape for o in ops_shapes), "Value shape mismatch.")

        elif isinstance(expression, Variable):
            shape = ops_shapes[0]
        
        elif isinstance(expression, Product):
            ufl_error("Not implemented!")
            
        elif isinstance(expression, Division):
            ufl_error("Not implemented!")
            
        elif isinstance(expression, Power):
            ufl_error("Not implemented!")

        elif isinstance(expression, (Mod, Abs, MathFunction)):
            shape = ()
            ufl_assert(ops_shapes[0] == (), "Non-scalar function argument.")
        
        elif isinstance(expression, Indexed):   
            ufl_error("Not implemented!") # FIXME

        elif isinstance(expression, PartialDerivative): 
            ufl_error("Not implemented!") # FIXME
            
        elif isinstance(expression, Diff): 
            ufl_error("Not implemented!") # FIXME
            
        elif isinstance(expression, Grad):
            shape = (dim,) + ops_shapes[0]
            
        elif isinstance(expression, Div):
            shape = ops_shapes[1:]
            ufl_assert(ops_shapes[0] == dim, "Value shape mismatch in Div.")        

        elif isinstance(expression, Curl): 
            ufl_error("Not implemented!") # FIXME

        elif isinstance(expression, Rot):  
            ufl_error("Not implemented!") # FIXME

        elif isinstance(expression, Outer): 
            ufl_error("Not implemented!") # FIXME
            
        elif isinstance(expression, Inner): 
            ufl_error("Not implemented!") # FIXME
            
        elif isinstance(expression, Dot):   
            ufl_error("Not implemented!") # FIXME
            
        elif isinstance(expression, Cross): 
            ufl_error("Not implemented!") # FIXME

        elif isinstance(expression, Trace):     
            ufl_error("Not implemented!") # FIXME
            
        elif isinstance(expression, Determinant): 
            ufl_error("Not implemented!") # FIXME

        elif isinstance(expression, Transpose): 
            ufl_error("Not implemented!") # FIXME
        
        elif isinstance(expression, Inverse):     
            ufl_error("Not implemented!") # FIXME
        
        elif isinstance(expression, Deviatoric):
            ufl_error("Not implemented!") # FIXME
        
        elif isinstance(expression, Cofactor):
            ufl_error("Not implemented!") # FIXME
        
        elif isinstance(expression, ListVector):
            ufl_error("Not implemented!") # FIXME
        
        elif isinstance(expression, ListMatrix):
            ufl_error("Not implemented!") # FIXME
        
        elif isinstance(expression, Tensor):
            ufl_error("Not implemented!") # FIXME
        
        else:
            ufl_error("Missing %s handler for computing value shape." % expression.__class__)

    return shape


def validate_form(a):
    """Performs all implemented validations on a form. Raises exception if something fails."""
    
    ufl_assert(isinstance(a, Form), "Expecting Form.")
    
    ufl_assert(is_multilinear(a), "Form is not multilinear.")
    
    for e in iter_expressions(a):
        shape = value_shape(e)
        ufl_assert(shape == (), "Got non-scalar integrand expression:\n%s" % e)
    
