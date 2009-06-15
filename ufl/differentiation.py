"Differential operators."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-05-05"

from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.common import subdict, mergedicts
from ufl.expr import Expr, Operator
from ufl.terminal import Terminal, Tuple
from ufl.constantvalue import ConstantValue, Zero, ScalarValue, Identity, is_true_ufl_scalar
from ufl.indexing import Index, FixedIndex, Indexed, as_multi_index, MultiIndex
from ufl.indexutils import unique_indices
from ufl.geometry import FacetNormal
from ufl.variable import Variable
from ufl.tensors import as_tensor, ComponentTensor, ListTensor
from ufl.function import Function, ConstantBase
from ufl.basisfunction import BasisFunction
from ufl.precedence import parstr

#--- Basic differentiation objects ---

def is_spatially_constant(expression):
    "Check if a terminal object is spatially constant, such that expression.dx(i) == 0."
    if isinstance(expression, (ConstantValue, ConstantBase)):
        return True
    elif isinstance(expression, FacetNormal) and expression.cell().degree() == 1:
        return True
    elif isinstance(expression, Function):
        e = expression.element()
        if e.family() == "Discontinuous Lagrange" and e.degree() == 0: # TODO: Only e.degree() == 0?
            return True
    elif isinstance(expression, ListTensor):
        return all(is_spatially_constant(e) for e in expression.operands())
    elif isinstance(expression, (Indexed, ComponentTensor)):
        return is_spatially_constant(expression.operands()[0])
    
    return False

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
        return "FunctionDerivative (w.r.t. function %s and using basis function %s) of \n%s" % (self._functions, self._basis_functions, self._integrand) # TODO: Short notation
    
    def __repr__(self):
        return "FunctionDerivative(%r, %r, %r)" % (self._integrand, self._functions, self._basis_functions)

def split_indices(expression, idx):
    idims = dict(expression.index_dimensions())
    if isinstance(idx, Index) and idims.get(idx) is None:
        cell = expression.cell()
        ufl_assert(cell is not None,
            "Need to know the spatial dimension to "\
            "compute the shape of derivatives.")
        idims[idx] = cell.geometric_dimension()
    fi = unique_indices(expression.free_indices() + (idx,))
    return fi, idims

class SpatialDerivative(Derivative):
    "Partial derivative of an expression w.r.t. spatial directions given by indices."
    __slots__ = ("_expression", "_index", "_shape", "_free_indices", "_index_dimensions", "_repr")
    def __new__(cls, expression, index):
        # Return zero if expression is trivially constant
        if is_spatially_constant(expression):
            if isinstance(index, (tuple, MultiIndex)):
                index, = index
            fi, idims = split_indices(expression, index)
            return Zero(expression.shape(), fi, idims)
        return Derivative.__new__(cls)
    
    def __init__(self, expression, index):
        Derivative.__init__(self)
        self._expression = expression
        
        # Make a MultiIndex with knowledge of the dimensions
        cell = expression.cell()
        if cell is None:
            sh = None
            error("Cannot compute derivatives of expressions with no cell.")
        else:
            sh = (cell.geometric_dimension(),)
        self._index = as_multi_index(index, sh)

        # Make sure we have a single valid index
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
        if isinstance(self._expression, Terminal):
            return "d%s/dx_%s" % (self._expression, self._index)
        return "d/dx_%s %s" % (self._index, parstr(self._expression, self))
    
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
            return Zero(f.shape() + v.shape(), free_indices, index_dimensions)
            #return Zero(v.shape() + f.shape(), free_indices, index_dimensions) # DIFFSHAPE TODO: Use this version instead?
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
        #print "set(fi)", set(fi)
        #print "set(vi)", set(vi)
        #print "^", (set(fi) ^ set(vi))
        ufl_assert(not (set(fi) & set(vi)), \
            "Repeated indices not allowed in VariableDerivative.") # TODO: Allow diff(f[i], v[i]) = sum_i VariableDerivative(f[i], v[i])? Can implement direct expansion in diff as a first approximation.
        self._free_indices = tuple(fi + vi)
        self._index_dimensions = dict(fid)
        self._index_dimensions.update(vid)
        self._shape = f.shape() + v.shape()
        #self._shape = v.shape() + f.shape() # DIFFSHAPE TODO: Use this version instead?
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
        if isinstance(self._f, Terminal):
            return "d%s/d[%s]" % (self._f, self._v)
        return "d/d[%s] %s" % (self._v, parstr(self._f, self))

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
            dim = cell.geometric_dimension()
            free_indices = f.free_indices()
            index_dimensions = subdict(f.index_dimensions(), free_indices)
            return Zero((dim,) + f.shape(), free_indices, index_dimensions)
        return CompoundDerivative.__new__(cls)
    
    def __init__(self, f):
        CompoundDerivative.__init__(self)
        self._f = f
        cell = f.cell()
        ufl_assert(cell is not None, "Can't take gradient of expression with undefined cell. How did this happen?")
        self._dim = cell.geometric_dimension()
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
        sh = f.shape()
        ufl_assert(f.shape() in ((), (2,), (3,)), "Expecting a scalar, 2D vector or 3D vector.")
        ufl_assert(not f.free_indices(), \
            "TODO: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")
        
        # Return zero if expression is trivially constant
        if is_spatially_constant(f):
            cell = f.cell()
            ufl_assert(cell is not None, "Can't take curl of expression with undefined cell...")
            sh = { (): (2,), (2,): (), (3,): (3,) }[sh]
            #free_indices = f.free_indices()
            #index_dimensions = subdict(f.index_dimensions(), free_indices)
            #return Zero((cell.geometric_dimension(),), free_indices, index_dimensions)
            return Zero(sh)
        return CompoundDerivative.__new__(cls)
    
    def __init__(self, f):
        CompoundDerivative.__init__(self)
        cell = f.cell()
        ufl_assert(cell is not None, "Can't take curl of expression with undefined cell...")
        sh = { (): (2,), (2,): (), (3,): (3,) }[f.shape()]
        self._f = f
        self._shape = sh
        self._repr = "Curl(%r)" % self._f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def index_dimensions(self):
        return self._f.index_dimensions()
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "curl(%s)" % self._f
    
    def __repr__(self):
        return self._repr

