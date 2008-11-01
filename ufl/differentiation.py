"Differential operators."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-11-01"

from .output import ufl_assert
from .base import Expr, Terminal
from .zero import Zero
from .scalar import ScalarValue
from .indexing import Indexed, MultiIndex, Index, DefaultDim, extract_indices
from .variable import Variable
from .tensors import as_tensor
from .tensoralgebra import Identity
from .function import Function, Constant, VectorConstant, TensorConstant

#--- Basic differentiation objects ---

spatially_constant_types = (ScalarValue, Zero, Identity, Constant, VectorConstant, TensorConstant) # FacetNormal: not for higher order geometry!

class SpatialDerivative(Expr):
    "Partial derivative of an expression w.r.t. spatial directions given by indices."
    __slots__ = ("_expression", "_shape", "_indices", "_free_indices", "_repeated_indices", "_dx_free_indices", "_dx_repeated_indices")
    def __new__(cls, expression, indices):
        if isinstance(expression, Terminal):
            # Return zero if expression is trivially 
            # constant, and there are no free indices.
            # (there are no expression.free_indices() in terminal types)
            ind = [i for i in indices if isinstance(i, Index)]
            si = set(ind)
            if len(ind) == 2*len(si):
                if isinstance(expression, spatially_constant_types):
                    return Zero(expression.shape())
        return Expr.__new__(cls)
    
    def __init__(self, expression, indices):
        self._expression = expression
        
        if not isinstance(indices, MultiIndex):
            # if constructed from repr
            indices = MultiIndex(indices, len(indices)) # TODO: Do we need len(indices) in MultiIndex?
        self._indices = indices
        
        # Find free and repeated indices in the dx((i,i,j)) part
        (self._dx_free_indices, self._dx_repeated_indices, dummy) = \
            extract_indices(self._indices._indices)
        
        # Find free and repeated indices among the combined
        # indices of the expression and dx((i,j,k))
        indices = expression.free_indices() + self._dx_free_indices
        (self._free_indices, self._repeated_indices, self._shape) = \
            extract_indices(indices, expression.shape())
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices

    def repeated_indices(self):
        return self._repeated_indices
    
    def shape(self):
        return self._shape

    def repeated_index_dimensions(self, default_dim):
        # Repeated indices here always iterate over the default
        # spatial range, so I think this should be correct:
        d = {}
        for i in self._repeated_indices:
            d[i] = default_dim
        return d
    
    def __str__(self):
        # TODO: Pretty-print for higher order derivatives.
        return "(d[%s] / dx_%s)" % (self._expression, self._indices)
    
    def __repr__(self):
        return "SpatialDerivative(%r, %r)" % (self._expression, self._indices)

class VariableDerivative(Expr):
    __slots__ = ("_f", "_v", "_index", "_free_indices", "_shape")
    def __new__(cls, f, v):
        # Return zero if expression is trivially independent 
        # of Function, and there are no free indices
        if (not isinstance(f, Variable)) and isinstance(f, Terminal):
            # Remove repeated indices to get the free 
            free_indices = set(f.free_indices()) ^ set(v.free_indices())
            if not free_indices:
                return Zero(f.shape())
        return Expr.__new__(cls)
    
    def __init__(self, f, v):
        ufl_assert(isinstance(f, Expr), "Expecting an Expr in VariableDerivative.")
        if isinstance(v, Indexed):
            ufl_assert(isinstance(v._expression, Variable), \
                "Expecting a Variable in VariableDerivative.")
            ufl_warning("diff(f, v[i]) probably isn't handled properly in all code.") # FIXME
        else:
            ufl_assert(isinstance(v, Variable), \
                "Expecting a Variable in VariableDerivative.")
        self._f = f
        self._v = v
        fi = f.free_indices()
        vi = v.free_indices()
        ufl_assert(len(set(fi) ^ set(vi)) == 0, \
            "Repeated indices not allowed in VariableDerivative.") # TODO: Allow diff(f[i], v[i])?
        self._free_indices = tuple(fi + vi)
        self._shape = f.shape() + v.shape()
    
    def operands(self):
        return (self._f, self._v)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "(d[%s] / d[%s])" % (self._f, self._v)

    def __repr__(self):
        return "VariableDerivative(%r, %r)" % (self._f, self._v)

#--- Compound differentiation objects ---

class Grad(Expr):
    __slots__ = ("_f",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if isinstance(f, spatially_constant_types):
            return Zero((DefaultDim,) + f.shape())
        return Expr.__new__(cls)
    
    def __init__(self, f):
        self._f = f
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking gradient of an expression with free indices, should this be a valid expression? Please provide examples!")
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def shape(self):
        return (DefaultDim,) + self._f.shape()
    
    def __str__(self):
        return "grad(%s)" % self._f
    
    def __repr__(self):
        return "Grad(%r)" % self._f

class Div(Expr):
    __slots__ = ("_f",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if isinstance(f, spatially_constant_types):
            return Zero(f.shape()[1:])
        return Expr.__new__(cls)

    def __init__(self, f):
        ufl_assert(f.rank() >= 1, "Can't take the divergence of a scalar.")
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking divergence of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def shape(self):
        return self._f.shape()[1:]
    
    def __str__(self):
        return "div(%s)" % self._f

    def __repr__(self):
        return "Div(%r)" % self._f

class Curl(Expr):
    __slots__ = ("_f",)
    
    def __init__(self, f):
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def shape(self):
        return (DefaultDim,)
    
    def __str__(self):
        return "curl(%s)" % self._f
    
    def __repr__(self):
        return "Curl(%r)" % self._f

class Rot(Expr):
    __slots__ = ("_f",)

    def __init__(self, f):
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking rot of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "rot(%s)" % self._f
    
    def __repr__(self):
        return "Rot(%r)" % self._f
