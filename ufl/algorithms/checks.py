"""Functions to check the validity of forms."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-13"

from ..output import ufl_error, ufl_assert, ufl_info
# FIXME:
from ..all import UFLObject, Terminal, Number, Variable, Identity, FacetNormal
from ..all import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..all import BasisFunction, Function, Constant
from ..all import Indexed, MultiIndex
from ..all import ListVector, ListMatrix, Tensor
from ..all import Sum, Product, Division, Power, Mod, Abs, MathFunction
from ..all import Outer, Inner, Dot, Cross, Transpose, Inverse
from ..all import Trace, Determinant, Deviatoric, Cofactor
from ..all import PartialDerivative, Diff, Div, Grad, Curl, Rot

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
    
