"Differential operators."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-12-30"

from ufl.output import ufl_assert, ufl_warning
from ufl.common import subdict, mergedicts
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.zero import Zero
from ufl.scalar import ScalarValue
from ufl.indexing import Indexed, MultiIndex, Index, extract_indices
from ufl.variable import Variable
from ufl.tensors import as_tensor
from ufl.tensoralgebra import Identity
from ufl.function import Function, Constant, VectorConstant, TensorConstant

#--- Basic differentiation objects ---

spatially_constant_types = (ScalarValue, Zero, Identity, Constant, VectorConstant, TensorConstant) # FacetNormal: not for higher order geometry!

class Derivative(Expr):
    "Base class for all derivative types."
    __slots__ = ()
    def __init__(self):
        Expr.__init__(self)

class SpatialDerivative(Derivative):
    "Partial derivative of an expression w.r.t. spatial directions given by indices."
    __slots__ = ("_expression", "_shape", "_indices", "_free_indices", "_index_dimensions", "_repeated_indices", "_dx_free_indices", "_dx_repeated_indices")
    def __new__(cls, expression, indices):
        if isinstance(expression, Terminal):
            # Return zero if expression is trivially constant
            if isinstance(expression, spatially_constant_types):
                cell = expression.cell()
                ufl_assert(cell is not None, "Need cell to know spatial dimension in SpatialDerivative.")
                spatial_dim = cell.dim()
                
                # Compute free indices and their dimensions
                si = set(i for i in indices if isinstance(i, Index))
                free_indices = expression.free_indices() ^ si
                index_dimensions = dict(expression.index_dimensions())
                index_dimensions.update((i, spatial_dim) for i in si)
                index_dimensions = subdict(index_dimensions, free_indices)
                
                return Zero(expression.shape(), free_indices, index_dimensions)

        return Expr.__new__(cls)
    
    def __init__(self, expression, indices):
        Expr.__init__(self)
        self._expression = expression
        
        if not isinstance(indices, MultiIndex):
            # if constructed from repr
            indices = MultiIndex(indices, len(indices)) # TODO: Do we need len(indices) in MultiIndex?
        self._indices = indices
        
        # Find free and repeated indices in the dx((i,i,j)) part
        (self._dx_free_indices, self._dx_repeated_indices, dummy, dummy) = \
            extract_indices(self._indices._indices)
        
        cell = expression.cell()
        ufl_assert(cell is not None, "Need to know the spatial dimension to compute the shape of derivatives.")
        dim = cell.dim()
        self._index_dimensions = {}
        for i in self._dx_free_indices:
            # set free index dimensions to the spatial dimension 
            self._index_dimensions[i] = dim
        
        # Find free and repeated indices among the combined
        # indices of the expression and dx((i,j,k))
        fi = expression.free_indices()
        fid = expression.index_dimensions()
        indices = fi + self._dx_free_indices
        dimensions = tuple(fid[i] for i in fi) + (dim,)*len(self._dx_free_indices)
        (self._free_indices, self._repeated_indices, self._shape, self._index_dimensions) = \
            extract_indices(indices, dimensions)
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def repeated_indices(self):
        return self._repeated_indices
    
    def index_dimensions(self):
        return self._index_dimensions

    def shape(self):
        return self._shape

    def __str__(self):
        # TODO: Pretty-print for higher order derivatives.
        return "(d[%s] / dx_%s)" % (self._expression, self._indices)
    
    def __repr__(self):
        return "SpatialDerivative(%r, %r)" % (self._expression, self._indices)

class VariableDerivative(Derivative):
    __slots__ = ("_f", "_v", "_index", "_free_indices", "_index_dimensions", "_shape")
    def __new__(cls, f, v):
        # Return zero if expression is trivially independent of Function
        if isinstance(f, Terminal) and not isinstance(f, Variable):
            free_indices = set(f.free_indices()) ^ set(v.free_indices())
            index_dimensions = mergedicts([f.index_dimensions(), v.index_dimensions()])
            index_dimensions = subdict(index_dimensions, free_indices)
            return Zero(f.shape(), free_indices, index_dimensions)
        return Expr.__new__(cls)
    
    def __init__(self, f, v):
        Expr.__init__(self)
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
        fid = f.index_dimensions()
        vid = v.index_dimensions()
        ufl_assert(not (set(fi) ^ set(vi)), \
            "Repeated indices not allowed in VariableDerivative.") # TODO: Allow diff(f[i], v[i])?
        self._free_indices = tuple(fi + vi)
        self._index_dimensions = dict(fid)
        self._index_dimensions.update(vid)
        self._shape = f.shape() + v.shape()
    
    def operands(self):
        return (self._f, self._v)
    
    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "(d[%s] / d[%s])" % (self._f, self._v)

    def __repr__(self):
        return "VariableDerivative(%r, %r)" % (self._f, self._v)

#--- Compound differentiation objects ---

class Grad(Derivative):
    __slots__ = ("_f", "_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if isinstance(f, spatially_constant_types):
            cell = f.cell()
            ufl_assert(cell is not None, "Can't take gradient of expression with undefined cell...")
            dim = cell.dim()
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero((dim,) + f.shape(), free_indices, index_dimensions)
        return Expr.__new__(cls)
    
    def __init__(self, f):
        Expr.__init__(self)
        self._f = f
        cell = f.cell()
        ufl_assert(cell is not None, "Can't take gradient of expression with undefined cell. How did this happen?")
        self._dim = cell.dim()
        ufl_assert(not (f.free_indices()), \
            "TODO: Taking gradient of an expression with free indices, should this be a valid expression? Please provide examples!")
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def index_dimensions(self):
        return self._f.index_dimensions()
    
    def shape(self):
        return (self._dim,) + self._f.shape()
    
    def __str__(self):
        return "grad(%s)" % self._f
    
    def __repr__(self):
        return "Grad(%r)" % self._f

class Div(Derivative):
    __slots__ = ("_f",)

    def __new__(cls, f):
        ufl_assert(f.rank() >= 1, "Can't take the divergence of a scalar.")
        ufl_assert(not (f.free_indices()), \
            "TODO: Taking divergence of an expression with free indices, should this be a valid expression? Please provide examples!")
        # Return zero if expression is trivially constant
        if isinstance(f, spatially_constant_types):
            return Zero(f.shape()[1:]) # No free indices
        return Expr.__new__(cls)

    def __init__(self, f):
        Expr.__init__(self)
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def index_dimensions(self):
        return self._f.index_dimensions()
    
    def shape(self):
        return self._f.shape()[1:]
    
    def __str__(self):
        return "div(%s)" % self._f

    def __repr__(self):
        return "Div(%r)" % self._f

class Curl(Derivative):
    __slots__ = ("_f", "_dim",)
    
    # TODO: Implement __new__ to discover trivial zeros
    
    def __init__(self, f):
        Expr.__init__(self)
        ufl_assert(f.rank() == 1, "Need a vector.") # TODO: Is curl always 3D?
        ufl_assert(not f.free_indices(), \
            "TODO: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
        cell = f.cell()
        ufl_assert(cell is not None, "Can't take curl of expression with undefined cell...") # TODO: Is curl always 3D?
        self._dim = cell.dim()
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def index_dimensions(self):
        return self._f.index_dimensions()
    
    def shape(self):
        return (self._dim,)
    
    def __str__(self):
        return "curl(%s)" % self._f
    
    def __repr__(self):
        return "Curl(%r)" % self._f

class Rot(Derivative):
    __slots__ = ("_f",)
    
    # TODO: Implement __new__ to discover trivial zeros

    def __init__(self, f):
        Expr.__init__(self)
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(not f.free_indices(), \
            "TODO: Taking rot of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def index_dimensions(self):
        return self._f.index_dimensions()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "rot(%s)" % self._f
    
    def __repr__(self):
        return "Rot(%r)" % self._f
