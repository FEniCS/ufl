"Differential operators."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-03-25"

from ufl.log import warning
from ufl.assertions import ufl_assert
from ufl.common import subdict, mergedicts
from ufl.expr import Expr, Operator
from ufl.terminal import Terminal, Tuple
from ufl.constantvalue import ConstantValue, Zero, ScalarValue, Identity, is_true_ufl_scalar
from ufl.indexing import IndexBase, Index, FixedIndex, MultiIndex, Indexed, as_multi_index
from ufl.indexutils import unique_indices
from ufl.geometry import FacetNormal
from ufl.variable import Variable
from ufl.tensors import as_tensor
from ufl.function import Function, ConstantBase
from ufl.basisfunction import BasisFunction

#--- Basic differentiation objects ---

def is_spatially_constant(expression):
    return isinstance(expression, (ConstantValue, ConstantBase)) \
           or (isinstance(expression, FacetNormal) \
               and expression.cell().degree() == 1)

class Derivative(Operator):
    "Base class for all derivative types."
    __slots__ = ()
    def __init__(self):
        Operator.__init__(self)

class FunctionDerivative(Derivative):
    """Derivative of the integrand of a form w.r.t. the 
    degrees of freedom in a discrete Function."""
    __slots__ = ("_integrand", "_functions", "_basis_functions")
    
    def __new__(cls, integrand, functions, basis_functions):
        ufl_assert(is_true_ufl_scalar(integrand),
            "Expecting true UFL scalar expression.")
        ufl_assert(isinstance(functions, Tuple), #and all(isinstance(f, (Function,Indexed)) for f in functions),
            "Expecting Tuple instance with Functions.")
        ufl_assert(isinstance(basis_functions, Tuple), #and all(isinstance(f, BasisFunction) for f in basis_functions),
            "Expecting Tuple instance with BasisFunctions.")
        if isinstance(integrand, Zero):
            return Zero()
        return Derivative.__new__(cls)
    
    def __init__(self, integrand, functions, basis_functions):
        Derivative.__init__(self)
        self._integrand = integrand
        self._functions = functions
        self._basis_functions = basis_functions
    
    def operands(self):
        return (self._integrand, self._functions, self._basis_functions)
    
    def shape(self):
        return ()
    
    def free_indices(self):
        return ()
    
    def index_dimensions(self):
        return {}
    
    def __str__(self):
        return "FunctionDerivative (w.r.t. function %s and using basis function %s) of \n%s" % (self._functions, self._basis_functions, self._integrand)
    
    def __repr__(self):
        return "FunctionDerivative(%r, %r, %r)" % (self._integrand, self._functions, self._basis_functions)

def split_indices(expression, idx):
    idims = dict(expression.index_dimensions())
    if isinstance(idx, Index) and idims.get(idx) is None:
        cell = expression.cell()
        ufl_assert(cell is not None,
            "Need to know the spatial dimension to "\
            "compute the shape of derivatives.")
        dim = cell.d
        idims[idx] = dim
    fi = unique_indices(expression.free_indices() + (idx,))
    return fi, idims

class SpatialDerivative(Derivative):
    "Partial derivative of an expression w.r.t. spatial directions given by indices."
    __slots__ = ("_expression", "_index", "_shape", "_free_indices", "_index_dimensions", "_repr")
    def __new__(cls, expression, index):
        # Return zero if expression is trivially constant
        if is_spatially_constant(expression):
            index = as_multi_index(index)
            idx, = index
            fi, idims = split_indices(expression, idx)
            return Zero(expression.shape(), fi, idims)
        return Derivative.__new__(cls)
    
    def __init__(self, expression, index):
        Derivative.__init__(self)
        self._expression = expression
        
        # Make sure we have a single valid index
        self._index = as_multi_index(index)
        ufl_assert(len(self._index) == 1, "Expecting a single index.")
        fi, idims = split_indices(expression, self._index[0])
        
        # Store what we need
        self._free_indices = fi
        self._index_dimensions = idims
        self._shape = expression.shape()

        self._repr = "SpatialDerivative(%r, %r)" % (self._expression, self._index)
    
    def operands(self):
        return (self._expression, self._index)
   
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._shape
    
    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        
        i = self._index[0]

        if isinstance(i, FixedIndex):
            i = int(i)
        
        pushed = False
        if isinstance(i, Index):
            i = index_values[i]
            pushed = True
            index_values.push(self._index[0], None)
        
        result = self._expression.evaluate(x, mapping, component, index_values, derivatives=derivatives + (i,))
        
        if pushed:
            index_values.pop()
        
        return result
    
    def __str__(self):
        # TODO: Pretty-print for higher order derivatives.
        return "(d[%s] / dx_%s)" % (self._expression, self._index)
    
    def __repr__(self):
        return self._repr

class VariableDerivative(Derivative):
    __slots__ = ("_f", "_v", "_free_indices", "_index_dimensions", "_shape", "_repr")
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
            error("diff(f, v[i]) isn't handled properly in all code.") # TODO: Should we allow this? Can do diff(f, v)[..., i], which leads to additional work but does the same.
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
            "Repeated indices not allowed in VariableDerivative.") # TODO: Allow diff(f[i], v[i]) = sum_i VariableDerivative(f[i], v[i])? Can implement direct expansion in diff as a first approximation.
        self._free_indices = tuple(fi + vi)
        self._index_dimensions = dict(fid)
        self._index_dimensions.update(vid)
        self._shape = f.shape() + v.shape()
        self._repr = "VariableDerivative(%r, %r)" % (self._f, self._v)
    
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
        return self._repr

#--- Compound differentiation objects ---

class CompoundDerivative(Derivative):
    "Base class for all compound derivative types."
    __slots__ = ()
    def __init__(self):
        Derivative.__init__(self)

class Grad(CompoundDerivative):
    __slots__ = ("_f", "_dim", "_repr")

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if is_spatially_constant(f):
            cell = f.cell()
            ufl_assert(cell is not None, "Can't take gradient of expression with undefined cell...")
            dim = cell.d
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero((dim,) + f.shape(), free_indices, index_dimensions)
        return CompoundDerivative.__new__(cls)
    
    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        cell = f.cell()
        ufl_assert(cell is not None, "Can't take gradient of expression with undefined cell. How did this happen?")
        self._dim = cell.d
        ufl_assert(not (f.free_indices()), \
            "TODO: Taking gradient of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._repr = "Grad(%r)" % self._f
    
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
        return self._repr

class Div(CompoundDerivative):
    __slots__ = ("_f", "_repr")

    def __new__(cls, f):
        ufl_assert(f.rank() >= 1, "Can't take the divergence of a scalar.")
        ufl_assert(not (f.free_indices()), \
            "TODO: Taking divergence of an expression with free indices, should this be a valid expression? Please provide examples!")
        # Return zero if expression is trivially constant
        if is_spatially_constant(f):
            return Zero(f.shape()[1:]) # No free indices
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        self._repr = "Div(%r)" % self._f
    
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
        return self._repr

class Curl(CompoundDerivative):
    __slots__ = ("_f", "_dim", "_repr")
    
    def __new__(cls, f):
        # Validate input
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(not f.free_indices(), \
            "TODO: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")
        
        # Return zero if expression is trivially constant
        if is_spatially_constant(f):
            cell = f.cell()
            ufl_assert(cell is not None, "Can't take curl of expression with undefined cell...")
            #free_indices = f.free_indices()
            #index_dimensions = subdict(f.index_dimensions(), free_indices)
            #return Zero((cell.d,), free_indices, index_dimensions)
            return Zero((cell.d,))
        return CompoundDerivative.__new__(cls)
    
    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        cell = f.cell()
        ufl_assert(cell is not None, "Can't take curl of expression with undefined cell...")
        self._dim = cell.d
        self._repr = "Curl(%r)" % self._f
    
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
        return self._repr

class Rot(CompoundDerivative):
    __slots__ = ("_f", "_repr")
    
    def __new__(cls, f):
        # Validate input
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(not f.free_indices(), \
            "TODO: Taking rot of an expression with free indices, should this be a valid expression? Please provide examples!")
        
        # Return zero if expression is trivially constant
        if is_spatially_constant(f):
            #cell = f.cell()
            #ufl_assert(cell is not None, "Can't take rot of expression with undefined cell...")
            #free_indices = f.free_indices()
            #index_dimensions = subdict(f.index_dimensions(), free_indices)
            #return Zero((cell.d,), free_indices, index_dimensions)
            return Zero()
        return CompoundDerivative.__new__(cls)
    
    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        self._repr = "Rot(%r)" % self._f
    
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
        return self._repr
