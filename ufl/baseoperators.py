"""This module attaches special functions to UFLObject.
This way we avoid circular dependencies between e.g.
Sum and its superclass UFLObject."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-18 -- 2008-08-20"

# UFL imports
from .output import ufl_error
from .base import UFLObject, Number, ZeroType, is_python_scalar
from .algebra import Sum, Product, Division, Power, Mod, Abs
from .tensoralgebra import Transposed
from .indexing import Indexed
from .restriction import PositiveRestricted, NegativeRestricted
from .differentiation import PartialDerivative


#--- Extend UFLObject with algebraic operators ---

def _add(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    if isinstance(o, ZeroType): return self
    if isinstance(self, ZeroType): return o
    return Sum(self, o)
UFLObject.__add__ = _add

def _radd(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    if isinstance(o, ZeroType): return self
    if isinstance(self, ZeroType): return o
    return Sum(o, self)
UFLObject.__radd__ = _radd

def _sub(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    if isinstance(o, ZeroType): return self
    if isinstance(self, ZeroType): return -o
    return self + (-o)
UFLObject.__sub__ = _sub

def _rsub(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    if isinstance(self, ZeroType): return o
    if isinstance(o, ZeroType): return -self
    return o + (-self)
UFLObject.__rsub__ = _rsub

def _mul(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Product(self, o)
UFLObject.__mul__ = _mul

def _rmul(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Product(o, self)
UFLObject.__rmul__ = _rmul

def _div(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Division(self, o)
UFLObject.__div__ = _div

def _rdiv(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Division(o, self)
UFLObject.__rdiv__ = _rdiv

def _pow(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Power(self, o)
UFLObject.__pow__ = _pow

def _rpow(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Power(o, self)
UFLObject.__rpow__ = _rpow

def _mod(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Mod(self, o)
UFLObject.__mod__ = _mod

def _rmod(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Mod(o, self)
UFLObject.__rmod__ = _rmod

def _neg(self):
    return -1*self
UFLObject.__neg__ = _neg

def _abs(self):
    return Abs(self)
UFLObject.__abs__ = _abs

#--- Extend UFLObject with indexing operator a[i] ---

def _getitem(self, key):
    return Indexed(self, key)
UFLObject.__getitem__ = _getitem

#--- Extend UFLObject with restiction operators a("+"), a("-") ---

def _restrict(self, side):
    if side == "+":
        return PositiveRestricted(self)
    if side == "-":
        return NegativeRestricted(self)
    ufl_error("Invalid side %r in restriction operator." % side)
UFLObject.__call__ = _restrict

#--- Extend UFLObject with the transpose operation A.T ---

def _transpose(self):
    """Transposed a rank two tensor expression. For more general transpose
    operations of higher order tensor expressions, use indexing and Tensor."""
    return Transposed(self)
UFLObject.T = property(_transpose)

#--- Extend UFLObject with spatial differentiation operator a.dx(i) ---

def _dx(self, *i):
    """Return the partial derivative with respect to spatial variable number i"""
    return PartialDerivative(self, i)
UFLObject.dx = _dx

