"Differential operators."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-01-16"

from ufl.log import ufl_assert, warning
from ufl.common import subdict, mergedicts
from ufl.expr import Expr
from ufl.terminal import Terminal, Tuple
from ufl.zero import Zero
from ufl.scalar import ScalarValue, is_true_ufl_scalar
from ufl.indexing import Indexed, MultiIndex, Index, FixedIndex, extract_indices_for_dx
from ufl.variable import Variable
from ufl.tensors import as_tensor
from ufl.tensoralgebra import Identity
from ufl.function import Function, Constant, VectorConstant, TensorConstant
from ufl.basisfunction import BasisFunction

#--- Basic differentiation objects ---

spatially_constant_types = (ScalarValue, Zero, Identity, Constant, VectorConstant, TensorConstant) # FacetNormal: not for higher order geometry!

class Derivative(Expr):
    "Base class for all derivative types."
    __slots__ = ()
    def __init__(self):
        Expr.__init__(self)

class FunctionDerivative(Derivative):
    """Derivative of the integrand of a form w.r.t. the 
    degrees of freedom in a discrete Function."""
    __slots__ = ("_integrand", "_functions", "_basisfunctions")
    
    def __new__(cls, integrand, functions, basisfunctions):
        ufl_assert(is_true_ufl_scalar(integrand),
            "Expecting true UFL scalar expression.")
        ufl_assert(isinstance(functions, Tuple), #and all(isinstance(f, (Function,Indexed)) for f in functions),
            "Expecting Tuple instance with Functions.")
        ufl_assert(isinstance(basisfunctions, Tuple), #and all(isinstance(f, BasisFunction) for f in basisfunctions),
            "Expecting Tuple instance with BasisFunctions.")
        if isinstance(integrand, Zero):
            return Zero()
        return Derivative.__new__(cls)
    
    def __init__(self, integrand, functions, basisfunctions):
        Derivative.__init__(self)
        self._integrand = integrand
        self._functions = functions
        self._basisfunctions = basisfunctions
    
    def operands(self):
        return (self._integrand, self._functions, self._basisfunctions)
    
    def shape(self):
        return ()
    
    def free_indices(self):
        return ()
    
    def index_dimensions(self):
        return {}
    
    def __str__(self):
        return "FunctionDerivative (w.r.t. function %s and using basis function %s) of \n%s" % (self._functions, self._basisfunctions, self._integrand)
    
    def __repr__(self):
        return "FunctionDerivative(%r, %r, %r)" % (self._integrand, self._functions, self._basisfunctions)

class SpatialDerivative(Derivative):
    "Partial derivative of an expression w.r.t. spatial directions given by indices."
    __slots__ = ("_expression", "_index", "_shape", "_free_indices", "_repeated_indices", "_index_dimensions")
    def __new__(cls, expression, index):
        
        # Make sure we have a single valid index
        if isinstance(index, int):
            index = FixedIndex(index)
        if isinstance(index, (FixedIndex, Index)):
            index = MultiIndex((index,))
        ufl_assert(isinstance(index, MultiIndex), "Expecting a valid index type.")
        ufl_assert(len(index) == 1, "Can only handle one differentiation index!")
        idx = index[0]
        
        # Return zero if expression is trivially constant
        if isinstance(expression, Terminal):
            # TODO: Check for BasisFunction or Function and attach indices to them directly?
            if isinstance(expression, spatially_constant_types):
                # TODO: This code is duplicated in __init__
                # Get spatial dimension
                cell = expression.cell()
                if cell is None:
                    dim = None
                else:
                    dim = cell.dim()
                
                fi, ri, idim = extract_indices_for_dx(expression, idx)
                
                return Zero(expression.shape(), fi, idim)

        return Derivative.__new__(cls)
    
    def __init__(self, expression, index):
        Derivative.__init__(self)
        self._expression = expression
        
        # Make sure we have a single valid index
        if isinstance(index, int):
            index = FixedIndex(index)
        if isinstance(index, (FixedIndex, Index)):
            index = MultiIndex((index,))
        ufl_assert(isinstance(index, MultiIndex), "Expecting a valid index type.")
        ufl_assert(len(index) == 1, "Can only handle one differentiation index!")
        
        idx = index[0]
        self._index = index
        
        # TODO: This code is duplicated in __new__
        # Get spatial dimension
        cell = expression.cell()
        if cell is None:
            dim = None
        else:
            dim = cell.dim()
        
        fi, ri, idim = extract_indices_for_dx(expression, idx)
        
        # Store what we need
        self._free_indices = fi
        self._repeated_indices = ri
        self._index_dimensions = idim
        self._shape = expression.shape()
    
    def operands(self):
        return (self._expression, self._index)
   
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
        return "(d[%s] / dx_%s)" % (self._expression, self._index)
    
    def __repr__(self):
        return "SpatialDerivative(%r, %r)" % (self._expression, self._index)

class VariableDerivative(Derivative):
    __slots__ = ("_f", "_v", "_free_indices", "_index_dimensions", "_shape")
    def __new__(cls, f, v):
        # Return zero if expression is trivially independent of Function
        if isinstance(f, Terminal):# and not isinstance(f, Variable):
            free_indices = set(f.free_indices()) ^ set(v.free_indices())
            index_dimensions = mergedicts([f.index_dimensions(), v.index_dimensions()])
            index_dimensions = subdict(index_dimensions, free_indices)
            return Zero(f.shape(), free_indices, index_dimensions)
        return Derivative.__new__(cls)
    
    def __init__(self, f, v):
        Derivative.__init__(self)
        ufl_assert(isinstance(f, Expr), "Expecting an Expr in VariableDerivative.")
        if isinstance(v, Indexed):
            ufl_assert(isinstance(v._expression, Variable), \
                "Expecting a Variable in VariableDerivative.")
            warning("diff(f, v[i]) isn't handled properly in all code.") # FIXME
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
        return Derivative.__new__(cls)
    
    def __init__(self, f):
        Derivative.__init__(self)
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
        return Derivative.__new__(cls)

    def __init__(self, f):
        Derivative.__init__(self)
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
        Derivative.__init__(self)
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
        Derivative.__init__(self)
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
