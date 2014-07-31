"This module defines classes representing constant values."

# Copyright (C) 2008-2014 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2011.

from six.moves import zip
from six.moves import xrange as range

from ufl.log import warning, error
from ufl.assertions import ufl_assert, expecting_python_scalar
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.indexing import Index, FixedIndex
from ufl.common import EmptyDict
from ufl.core.ufl_type import ufl_type

#--- Helper functions imported here for compatibility---
from ufl.checks import is_python_scalar, is_ufl_scalar, is_true_ufl_scalar

# Precision for float formatting
precision = None
def format_float(x):
    "Format float value based on global UFL precision."
    if precision is None:
        return repr(x)
    else:
        return ("%%.%dg" % precision) % x

#--- Base classes for constant types ---

@ufl_type(is_abstract=True)
class ConstantValue(Terminal):
    __slots__ = ()
    def __init__(self):
        Terminal.__init__(self)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return True

    def domains(self):
        "Return tuple of domains related to this terminal object."
        return ()

@ufl_type(is_abstract=True)
class IndexAnnotated(ConstantValue):
    """Class to annotate expressions that don't depend on
    indices with a set of free indices, used internally to keep
    index properties intact during automatic differentiation."""

    __slots__ = ("ufl_shape", "_free_indices", "_index_dimensions")

    def __init__(self, shape=(), free_indices=(), index_dimensions=None):
        ConstantValue.__init__(self)
        if not all(isinstance(i, int) for i in shape):
            error("Expecting tuple of int.")
        if not all(isinstance(i, Index) for i in free_indices):
            error("Expecting tuple of Index objects.")
        self.ufl_shape = shape
        self._free_indices = tuple(sorted(free_indices, key=lambda x: x.count()))
        self._index_dimensions = dict(index_dimensions) if index_dimensions else EmptyDict
        if (set(self._free_indices) ^ set(self._index_dimensions.keys())):
            error("Index set mismatch.")

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

#--- Class for representing abstract constant symbol only for use internally in form compilers
#@ufl_type()
#class AbstractSymbol(ConstantValue):
#    "UFL literal type: Representation of a constant valued symbol with unknown properties."
#    __slots__ = ("_name", "ufl_shape")
#    def __init__(self, name, shape):
#        ConstantValue.__init__(self)
#        self._name = name
#        self.ufl_shape = shape
#
#    def reconstruct(self, name=None):
#        if name is None:
#            name = self._name
#        return AbstractSymbol(name, self.ufl_shape)
#
#    def __str__(self):
#        return "<Abstract symbol named '%s' with shape %s>" % (self._name, self.ufl_shape)
#
#    def __repr__(self):
#        return "AbstractSymbol(%r, %r)" % (self._name, self.ufl_shape)
#
#    def __eq__(self, other):
#        return isinstance(other, AbstractSymbol) and self._name == other._name and self.ufl_shape == other.ufl_shape


#--- Class for representing zero tensors of different shapes ---

# TODO: Add geometric dimension and Argument dependencies to Zero?
@ufl_type()
class Zero(IndexAnnotated):
    "UFL literal type: Representation of a zero valued expression."
    __slots__ = ()
    _cache = {}
    def __new__(cls, shape=(), free_indices=(), index_dimensions=None):
        if free_indices:
            self = IndexAnnotated.__new__(cls)
        else:
            self = Zero._cache.get(shape)
            if self is None:
                self = IndexAnnotated.__new__(cls)
                Zero._cache[shape] = self
        return self

    def __init__(self, shape=(), free_indices=(), index_dimensions=None):
        if not hasattr(self, '_shape'):
            ufl_assert(isinstance(free_indices, tuple),
                       "Expecting tuple of free indices, not %s" % str(free_indices))
            IndexAnnotated.__init__(self, shape, free_indices, index_dimensions)

    def reconstruct(self, free_indices=None):
        if not free_indices:
            return self
        ufl_assert(len(free_indices) == len(self._free_indices), "Size mismatch between old and new indices.")
        new_index_dimensions = dict((b, self._index_dimensions[a]) for (a, b) in zip(self._free_indices, free_indices))
        return Zero(self.ufl_shape, free_indices, new_index_dimensions)

    def __getnewargs__(self):
        return (self.ufl_shape, self._free_indices, self._index_dimensions)

    def evaluate(self, x, mapping, component, index_values):
        return 0.0

    def __str__(self):
        if self.ufl_shape == () and self._free_indices == ():
            return "0"
        return "(0<%r, %r>)" % (self.ufl_shape, self._free_indices)

    def __repr__(self):
        return "Zero(%r, %r, %r)" % (self.ufl_shape,
                self._free_indices, self._index_dimensions)

    def __eq__(self, other):
        if not isinstance(other, Zero):
            return isinstance(other, (int, float)) and other == 0
        if self is other:
            return True
        return (self.ufl_shape == other.ufl_shape and
                self._free_indices == other._free_indices and
                self._index_dimensions == other._index_dimensions)

    def __neg__(self):
        return self

    def __abs__(self):
        return self

    def __bool__(self):
        return False
    __nonzero__ = __bool__

    def __float__(self):
        return 0.0

    def __int__(self):
        return 0

def zero(*shape):
    "UFL literal constant: Return a zero tensor with the given shape."
    if len(shape) == 1 and isinstance(shape[0], tuple):
        return Zero(shape[0])
    else:
        return Zero(shape)

#--- Scalar value types ---

@ufl_type(is_abstract=True)
class ScalarValue(IndexAnnotated):
    "A constant scalar value."
    __slots__ = ("_value",)

    def __new__(cls, value, shape=(), free_indices=(), index_dimensions=None):
        is_python_scalar(value) or expecting_python_scalar(value)
        if value == 0:
            return Zero(shape, free_indices, index_dimensions)
        return IndexAnnotated.__new__(cls)

    def __getnewargs__(self):
        return (self._value, self.ufl_shape, self._free_indices, self._index_dimensions)

    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        IndexAnnotated.__init__(self, shape, free_indices, index_dimensions)
        self._value = value

    def reconstruct(self, free_indices=None):
        "Reconstruct with new free indices."
        if not free_indices:
            return self
        ufl_assert(len(free_indices) == len(self._free_indices), "Size mismatch between old and new indices.")
        new_index_dimensions = dict((b, self._index_dimensions[a]) for (a, b) in zip(self._free_indices, free_indices))
        return self._ufl_class_(self._value, self.ufl_shape, free_indices, new_index_dimensions)

    def value(self):
        return self._value

    def evaluate(self, x, mapping, component, index_values):
        return self._value

    def __eq__(self, other):
        """This is implemented to allow comparison with python scalars.

        Note that this will make IntValue(1) != FloatValue(1.0),
        but ufl-python comparisons like
            IntValue(1) == 1.0
            FloatValue(1.0) == 1
        can still succeed. These will however not have the same
        hash value and therefore not collide in a dict."""
        if not isinstance(other, self._ufl_class_):
            return isinstance(other, (int, float)) and other == self._value
        else:
            return self._value == other._value

    def __str__(self):
        return str(self._value)

    def __float__(self):
        return float(self._value)

    def __int__(self):
        return int(self._value)

    def __neg__(self):
        return type(self)(-self._value)

    def __abs__(self):
        return type(self)(abs(self._value))

@ufl_type(wraps_type=float)
class FloatValue(ScalarValue):
    "UFL literal type: Representation of a constant scalar floating point value."
    __slots__ = ()
    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        ScalarValue.__init__(self,
                             float(value),
                             shape,
                             free_indices,
                             index_dimensions)

    def __repr__(self):
        return "%s(%s, %s, %s, %s)" % (type(self).__name__,
                                       format_float(self._value),
                                       repr(self.ufl_shape),
                                       repr(self._free_indices),
                                       repr(self._index_dimensions))

@ufl_type(wraps_type=int)
class IntValue(ScalarValue):
    "UFL literal type: Representation of a constant scalar integer value."
    __slots__ = ()
    _cache = {}
    def __new__(cls, value, shape=(), free_indices=(), index_dimensions=None):
        # Check if it is cache-able
        if shape or free_indices or index_dimensions or abs(value) > 100:
            self = ScalarValue.__new__(cls, value, shape, free_indices, index_dimensions)
        else:
            self = IntValue._cache.get(value)
            if self is None:
                self = ScalarValue.__new__(cls, value, shape, free_indices, index_dimensions)
                IntValue._cache[value] = self
        return self

    def __init__(self, value, shape=(), free_indices=(), index_dimensions=None):
        if not hasattr(self, '_value'):
            ScalarValue.__init__(self, int(value), shape, free_indices, index_dimensions)

    def __repr__(self):
        return "%s(%s, %s, %s, %s)" % (type(self).__name__, repr(self._value),
                                       repr(self.ufl_shape), repr(self._free_indices),
                                       repr(self._index_dimensions))

#--- Identity matrix ---

@ufl_type()
class Identity(ConstantValue):
    "UFL literal type: Representation of an identity matrix."
    __slots__ = ("_dim", "ufl_shape")

    def __init__(self, dim):
        ConstantValue.__init__(self)
        self._dim = dim
        self.ufl_shape = (dim, dim)

    def evaluate(self, x, mapping, component, index_values):
        a, b = component
        return 1 if a == b else 0

    def __getitem__(self, key):
        ufl_assert(len(key) == 2, "Size mismatch for Identity.")
        if all(isinstance(k, (int, FixedIndex)) for k in key):
            return IntValue(1) if (int(key[0]) == int(key[1])) else Zero()
        return Expr.__getitem__(self, key)

    def __str__(self):
        return "I"

    def __repr__(self):
        return "Identity(%d)" % self._dim

    def __eq__(self, other):
        return isinstance(other, Identity) and self._dim == other._dim

#--- Permutation symbol ---

@ufl_type()
class PermutationSymbol(ConstantValue):
    """UFL literal type: Representation of a permutation symbol.

    This is also known as the Levi-Civita symbol, antisymmetric symbol,
    or alternating symbol."""
    __slots__ = ("ufl_shape", "_dim")

    def __init__(self, dim):
        ConstantValue.__init__(self)
        self._dim = dim
        self.ufl_shape = (dim,)*dim

    def evaluate(self, x, mapping, component, index_values):
        return self.__eps(component)

    def __getitem__(self, key):
        ufl_assert(len(key) == self._dim, "Size mismatch for PermutationSymbol.")
        if all(isinstance(k, (int, FixedIndex)) for k in key):
            return self.__eps(key)
        return Expr.__getitem__(self, key)

    def __str__(self):
        return "eps"

    def __repr__(self):
        return "PermutationSymbol(%d)" % self._dim

    def __eq__(self, other):
        return isinstance(other, PermutationSymbol) and self._dim == other._dim

    def __eps(self, x):
        """This function body is taken from
        http://www.mathkb.com/Uwe/Forum.aspx/math/29865/N-integer-Levi-Civita"""
        result = IntValue(1)
        for i, x1 in enumerate(x):
            for j in range(i + 1, len(x)):
                x2 = x[j]
                if x1 > x2:
                    result = -result
                elif x1 == x2:
                    return Zero()
        return result

def as_ufl(expression):
    "Converts expression to an Expr if possible."
    if isinstance(expression, Expr):
        return expression
    if isinstance(expression, int):
        return IntValue(expression)
    if isinstance(expression, float):
        return FloatValue(expression)
    error(("Invalid type conversion: %s can not be converted to any UFL type.\n"+\
           "The representation of the object is:\n%r") % (type(expression), expression))
