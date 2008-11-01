"""This module attaches special functions to Expr.
This way we avoid circular dependencies between e.g.
Sum and its superclass Expr."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-18 -- 2008-10-30"

# UFL imports
from .output import ufl_error, ufl_assert
from .base import Expr, Zero, ScalarValue, FloatValue, IntValue, is_python_scalar, as_ufl
from .algebra import Sum, Product, Division, Power, Abs
from .tensoralgebra import Transposed, Dot
from .indexing import Indexed
from .restriction import PositiveRestricted, NegativeRestricted
from .differentiation import SpatialDerivative


#--- Extend Expr with algebraic operators ---

from .base import _python_scalar_types
_valid_types = (Expr,) + _python_scalar_types

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
            return Zero(shape)
        return Dot(a, b)
        # TODO: Use index notation instead here?
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
    a = Indexed(self, key)
    if isinstance(self, Zero):
        return Zero(a.shape())
    return a
Expr.__getitem__ = _getitem

#--- Extend Expr with restiction operators a("+"), a("-") ---

def _restrict(self, side):
    if side == "+":
        return PositiveRestricted(self)
    if side == "-":
        return NegativeRestricted(self)
    ufl_error("Invalid side %r in restriction operator." % side)
Expr.__call__ = _restrict

#--- Extend Expr with the transpose operation A.T ---

def _transpose(self):
    """Transposed a rank two tensor expression. For more general transpose
    operations of higher order tensor expressions, use indexing and Tensor."""
    return Transposed(self)
Expr.T = property(_transpose)

#--- Extend Expr with spatial differentiation operator a.dx(i) ---

def _dx(self, *i):
    """Return the partial derivative with respect to spatial variable number i"""
    return SpatialDerivative(self, i)
Expr.dx = _dx
