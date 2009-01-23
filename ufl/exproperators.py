"""This module attaches special functions to Expr.
This way we avoid circular dependencies between e.g.
Sum and its superclass Expr."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-18 -- 2009-01-10"

# UFL imports
from ufl.log import error, ufl_assert
from ufl.common import subdict, mergedicts, StackDict
from ufl.expr import Expr
from ufl.zero import Zero
from ufl.scalar import ScalarValue, FloatValue, IntValue, is_python_scalar, as_ufl, python_scalar_types
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Transposed, Dot
from ufl.indexing import Indexed, as_index_tuple
from ufl.tensors import as_tensor
from ufl.restriction import PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative

#--- Extend Expr with algebraic operators ---

_valid_types = (Expr,) + python_scalar_types

def _add(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(self, o)
Expr.__add__ = _add

def _radd(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(o, self)
Expr.__radd__ = _radd

def _sub(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(self, -o)
Expr.__sub__ = _sub

def _rsub(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(o, -self)
Expr.__rsub__ = _rsub

def _mult(a, b):
    s1 = a.shape()
    s2 = b.shape()
    
    # Pick out valid non-scalar products here:
    # - matrix-matrix (A*B, M*grad(u)) => A . B
    # - matrix-vector (A*v) => A . v
    if len(s1) == 2 and (len(s2) == 2 or len(s2) == 1):
        shape = s1[:-1] + s2[1:]
        if isinstance(a, Zero) or isinstance(b, Zero):
            # Get free indices and their dimensions
            free_indices = tuple(set(a.free_indices()) ^ set(b.free_indices()))
            index_dimensions = mergedicts([a.free_index_dimensions(), b.free_index_dimensions()])
            index_dimensions = subdict(index_dimensions, free_indices)
            return Zero(shape, free_indices, index_dimensions)
        return Dot(a, b)
        # TODO: Use index notation instead here? If * is used in algorithms _after_ expand_compounds has been applied, returning Dot here may cause problems.
        #i = Index()
        #return a[...,i]*b[i,...]
    
    # Scalar products use Product:
    return Product(a, b)

def _mul(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    o = as_ufl(o)
    return _mult(self, o)
Expr.__mul__ = _mul

def _rmul(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    o = as_ufl(o)
    return _mult(o, self)
Expr.__rmul__ = _rmul

def _div(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Division(self, o)
Expr.__div__ = _div

def _rdiv(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Division(o, self)
Expr.__rdiv__ = _rdiv

def _pow(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Power(self, o)
Expr.__pow__ = _pow

def _rpow(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Power(o, self)
Expr.__rpow__ = _rpow

def _neg(self):
    return -1*self
Expr.__neg__ = _neg

def _abs(self):
    return Abs(self)
Expr.__abs__ = _abs

#--- Extend Expr with indexing operator a[i] ---

def _getitem(self, key):
    indices, axes = as_index_tuple(key)
    
    a = Indexed(self, indices)
    
    if isinstance(self, Zero):
        free_indices = a.free_indices()
        index_dimensions = subdict(a.index_dimensions(), free_indices)
        a = Zero(a.shape(), free_indices, index_dimensions)
    
    if axes: # TODO: what happens with both Zero and axes?
        a = as_tensor(a, axes)
    
    return a
Expr.__getitem__ = _getitem

#--- Extend Expr with restiction operators a("+"), a("-") ---

def _restrict(self, side):
    if side == "+":
        return PositiveRestricted(self)
    if side == "-":
        return NegativeRestricted(self)
    error("Invalid side %r in restriction operator." % side)
#Expr.__call__ = _restrict

def _call(self, arg, mapping=None):
    # Taking the restriction?
    if arg in ("+", "-"):
        ufl_assert(mapping is None, "Not expecting a mapping when taking restriction.")
        return _restrict(self, arg)
    
    # Evaluate expression at this particular coordinate,
    # with provided values for other terminals in mapping
    if mapping is None:
        mapping = {}
    component = ()
    index_values = StackDict()
    from ufl.algorithms import expand_derivatives
    f = expand_derivatives(self)
    return f.evaluate(arg, mapping, component, index_values)
Expr.__call__ = _call
    
#--- Extend Expr with the transpose operation A.T ---

def _transpose(self):
    """Transposed a rank two tensor expression. For more general transpose
    operations of higher order tensor expressions, use indexing and Tensor."""
    return Transposed(self)
Expr.T = property(_transpose)

#--- Extend Expr with spatial differentiation operator a.dx(i) ---

def _dx(self, *ii):
    "Return the partial derivative with respect to spatial variable number i."
    d = self
    for i in ii:
        d = SpatialDerivative(d, i)
    return d
Expr.dx = _dx

def _d(self, v):
    "Return the partial derivative with respect to variable v."
    return VariableDerivative(self, v)
Expr.d = _d

