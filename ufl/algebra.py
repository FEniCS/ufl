"""Basic algebra operations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-20 -- 2008-08-15"

# Modified by Anders Logg, 2008

from itertools import chain
from collections import defaultdict

from .output import ufl_assert, ufl_error
from .base import UFLObject, Number, is_true_scalar, is_python_scalar
from .indexing import extract_indices, compare_shapes

#--- Algebraic operators ---

class Sum(UFLObject):
    __slots__ = ("_operands",)
    
    def __init__(self, *operands):
        ufl_assert(len(operands), "Got sum of nothing.")
        s = operands[0].shape()
        
        ufl_assert(all(compare_shapes(s, o.shape()) for o in operands), "Shape mismatch in sum.")
        ufl_assert(all(operands[0].free_indices() == o.free_indices() for o in operands), "Can't add expressions with different free indices.")
        self._operands = tuple(operands)
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._operands[0].free_indices()
    
    def shape(self):
        return self._operands[0].shape()
    
    def __str__(self):
        return "(%s)" % " + ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " + ".join(repr(o) for o in self._operands)

class Product(UFLObject):
    __slots__ = ("_operands", "_shape", "_free_indices", "_repeated_indices")
    
    def __init__(self, *operands):
        ufl_assert(len(operands), "Got product of nothing.")
        self._operands = tuple(operands)
       
        # Extract indices
        all_indices = tuple(chain(*(o.free_indices() for o in operands)))
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = extract_indices(all_indices)
        self._free_indices       = free_indices
        self._repeated_indices   = repeated_indices

        # Try to determine shape of this sequence of
        # products with possibly varying shapes of each operand.
        # Products currently defined as valid are:
        # - something multiplied with a scalar
        # - a scalar multiplied with something
        # - matrix-matrix (A*B, M*grad(u))
        # - matrix-vector (A*v)
        # TODO: This logic is a bit shaky, feels too hacky, can we do more general?
        current_shape = operands[0].shape()
        for o in operands[1:]:
            o_shape = o.shape()
            if current_shape == () or o_shape == ():
                # at least one scalar
                current_shape = current_shape + o_shape
            elif len(current_shape) == 2 and len(o_shape) == 2:
                # matrix-matrix product
                current_shape = (current_shape[0], o_shape[1])
            elif len(current_shape) == 2 and len(o_shape) == 1:
                # matrix-vector product
                current_shape = (current_shape[0],)
            else:
                ufl_error("Invalid combination of tensor shapes in product.")
        self._shape = current_shape
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "(%s)" % " * ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " * ".join(repr(o) for o in self._operands)

class Division(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(b), "Division by non-scalar.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._a.free_indices()
    
    def shape(self):
        return self._a.shape()
    
    def __str__(self):
        return "(%s / %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%r / %r)" % (self._a, self._b)

class Power(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar power not defined.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return tuple()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "(%s ** %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%r ** %r)" % (self._a, self._b)

class Mod(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar mod not defined.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return tuple()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "(%s %% %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%r %% %r)" % (self._a, self._b)

class Abs(UFLObject):
    __slots__ = ("_a",)
    
    def __init__(self, a):
        self._a = a
    
    def operands(self):
        return (self._a, )
    
    def free_indices(self):
        return self._a.free_indices()
    
    def shape(self):
        return self._a.shape()
    
    def __str__(self):
        return "| %s |" % str(self._a)
    
    def __repr__(self):
        return "Abs(%r)" % self._a

#--- Extend UFLObject with algebraic operators ---

def _add(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Sum(self, o)
UFLObject.__add__ = _add

def _radd(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Sum(o, self)
UFLObject.__radd__ = _radd

def _sub(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return self + (-o)
UFLObject.__sub__ = _sub

def _rsub(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
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
