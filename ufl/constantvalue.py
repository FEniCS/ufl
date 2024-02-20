"""This module defines classes representing constant values."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2011.
# Modified by Massimiliano Leoni, 2016.

import typing
from math import atan2

import ufl
from ufl.checks import is_python_scalar, is_true_ufl_scalar, is_ufl_scalar  # noqa: F401
from ufl.core.expr import Expr
from ufl.core.multiindex import FixedIndex, Index
from ufl.core.terminal import Terminal

# Precision for float formatting
precision = None


def format_float(x):
    """Format float value based on global UFL precision."""
    if precision:
        return "{:.{prec}}".format(float(x), prec=precision)
    else:
        return "{}".format(float(x))


class ConstantValue(Terminal):
    """Constant value."""

    __slots__ = ("_shape",)

    def __init__(self, shape):
        """Initialise."""
        Terminal.__init__(self)
        self._shape = shape

    def _ufl_hash_data_(self):
        raise NotImplementedError()

    def __hash__(self):
        """Hash."""
        return hash(f"{self!r}")

    @property
    def ufl_shape(self):
        """UFL Shape."""
        return self._shape

    def is_cellwise_constant(self):
        """Return whether this expression is spatially constant over each cell."""
        return True

    def ufl_domains(self):
        """Return tuple of domains related to this terminal object."""
        return ()

    def apply_default_restrictions(self):
        """Apply default restrictions."""
        return self

    def apply_restrictions(self, side=None):
        """Apply restrictions."""
        return self


# TODO: Add geometric dimension/domain and Argument dependencies to Zero?
class Zero(ConstantValue):
    """Representation of a zero valued expression.

    Class for representing zero tensors of different shapes.
    """

    __slots__ = ("_ufl_free_indices", "_ufl_index_dimensions")

    _cache: typing.Dict[int, ConstantValue] = {}

    def __getnewargs__(self):
        """Get new args."""
        return (self.ufl_shape, self._ufl_free_indices, self._ufl_index_dimensions)

    def __new__(cls, shape=(), free_indices=(), index_dimensions=None):
        """Create new Zero."""
        if free_indices:
            self = ConstantValue.__new__(cls)
        else:
            self = Zero._cache.get(shape)
            if self is not None:
                return self
            self = ConstantValue.__new__(cls)
            Zero._cache[shape] = self
        self._init(shape, free_indices, index_dimensions)
        return self

    def __init__(self, shape=(), free_indices=(), index_dimensions=None):
        """Initialise."""
        pass

    def _init(self, shape=(), free_indices=(), index_dimensions=None):
        """Initialise."""
        ConstantValue.__init__(self, shape)

        if not all(isinstance(i, int) for i in shape):
            raise ValueError("Expecting tuple of int.")
        if not isinstance(free_indices, tuple):
            raise ValueError(f"Expecting tuple for free_indices, not {free_indices}.")

        if not free_indices:
            self._ufl_free_indices = ()
            self._ufl_index_dimensions = ()
        elif all(isinstance(i, Index) for i in free_indices):  # Handle old input format
            if not isinstance(index_dimensions, dict) and all(
                isinstance(i, Index) for i in index_dimensions.keys()
            ):
                raise ValueError(f"Expecting tuple of index dimensions, not {index_dimensions}")
            self._ufl_free_indices = tuple(sorted(i.count() for i in free_indices))
            self._ufl_index_dimensions = tuple(
                d for i, d in sorted(index_dimensions.items(), key=lambda x: x[0].count())
            )
        else:  # Handle new input format
            if not all(isinstance(i, int) for i in free_indices):
                raise ValueError(f"Expecting tuple of integer free index ids, not {free_indices}")
            if not isinstance(index_dimensions, tuple) and all(
                isinstance(i, int) for i in index_dimensions
            ):
                raise ValueError(
                    f"Expecting tuple of integer index dimensions, not {index_dimensions}"
                )

            # Assuming sorted now to avoid this cost, enable for debugging:
            # if sorted(free_indices) != list(free_indices):
            #    raise ValueError("Expecting sorted input. Remove this check later for efficiency.")

            self._ufl_free_indices = free_indices
            self._ufl_index_dimensions = index_dimensions

    @property
    def ufl_free_indices(self):
        """A tuple of free index counts."""
        return self._ufl_free_indices

    @property
    def ufl_index_dimensions(self):
        """A tuple providing the int dimension for each free index."""
        return self._ufl_index_dimensions

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        return 0.0

    def __str__(self):
        """Format as a string."""
        if self.ufl_shape == () and self._ufl_free_indices == ():
            return "0"
        if self._ufl_free_indices == ():
            return "0 (shape %s)" % (self.ufl_shape,)
        if self.ufl_shape == ():
            return "0 (index labels %s)" % (self._ufl_free_indices,)
        return "0 (shape %s, index labels %s)" % (self.ufl_shape, self._ufl_free_indices)

    def __repr__(self):
        """Representation."""
        r = "Zero(%s, %s, %s)" % (
            repr(self.ufl_shape),
            repr(self._ufl_free_indices),
            repr(self._ufl_index_dimensions),
        )
        return r

    def __eq__(self, other):
        """Check equalty."""
        if isinstance(other, Zero):
            if self is other:
                return True
            return (
                self.ufl_shape == other.ufl_shape
                and self._ufl_free_indices == other.ufl_free_indices
                and self._ufl_index_dimensions == other.ufl_index_dimensions
            )
        elif isinstance(other, (int, float)):
            return other == 0
        else:
            return False

    def __hash__(self):
        """Hash."""
        return super().__hash__()

    def __neg__(self):
        """Negate."""
        return self

    def __abs__(self):
        """Absolute value."""
        return self

    def __bool__(self):
        """Convert to a bool."""
        return False

    __nonzero__ = __bool__

    def __float__(self):
        """Convert to a float."""
        return 0.0

    def __int__(self):
        """Convert to an int."""
        return 0

    def __complex__(self):
        """Convert to a complex number."""
        return 0 + 0j


def zero(*shape):
    """UFL literal constant: Return a zero tensor with the given shape."""
    if len(shape) == 1 and isinstance(shape[0], tuple):
        return Zero(shape[0])
    else:
        return Zero(shape)


class ScalarValue(ConstantValue):
    """A constant scalar value."""

    __slots__ = ("_value",)

    def __init__(self, value):
        """Initialise."""
        ConstantValue.__init__(self, ())
        self._value = value

    def value(self):
        """Get the value."""
        return self._value

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        return self._value

    def __eq__(self, other):
        """Check equalty.

        This is implemented to allow comparison with python scalars.

        Note that this will make IntValue(1) != FloatValue(1.0),
        but ufl-python comparisons like
            IntValue(1) == 1.0
            FloatValue(1.0) == 1
        can still succeed. These will however not have the same
        hash value and therefore not collide in a dict.
        """
        if isinstance(other, self.__class__):
            return self._value == other._value
        elif isinstance(other, (int, float)):
            # FIXME: Disallow this, require explicit 'expr ==
            # IntValue(3)' instead to avoid ambiguities!
            return other == self._value
        else:
            return False

    def __hash__(self):
        """Hash."""
        return super().__hash__()

    def __str__(self):
        """Format as a string."""
        return str(self._value)

    def __float__(self):
        """Convert to a float."""
        return float(self._value)

    def __int__(self):
        """Convert to an int."""
        return int(self._value)

    def __complex__(self):
        """Convert to a complex number."""
        return complex(self._value)

    def __neg__(self):
        """Negate."""
        return type(self)(-self._value)

    def __abs__(self):
        """Absolute value."""
        return type(self)(abs(self._value))

    def real(self):
        """Real part."""
        return self._value.real

    def imag(self):
        """Imaginary part."""
        return self._value.imag


class ComplexValue(ScalarValue):
    """Representation of a constant, complex scalar."""

    __slots__ = ()

    def __getnewargs__(self):
        """Get new args."""
        return (self._value,)

    def __new__(cls, value):
        """Create a new ComplexValue."""
        if value.imag == 0:
            if value.real == 0:
                return Zero()
            else:
                return FloatValue(value.real)
        else:
            return ConstantValue.__new__(cls)

    def __init__(self, value):
        """Initialise."""
        ScalarValue.__init__(self, complex(value))

    def modulus(self):
        """Get the modulus."""
        return abs(self.value())

    def argument(self):
        """Get the argument."""
        return atan2(self.value().imag, self.value().real)

    def __repr__(self):
        """Representation."""
        r = "%s(%s)" % (type(self).__name__, repr(self._value))
        return r

    def __float__(self):
        """Convert to a float."""
        raise TypeError("ComplexValues cannot be cast to float")

    def __int__(self):
        """Convert to an int."""
        raise TypeError("ComplexValues cannot be cast to int")


class RealValue(ScalarValue):
    """Abstract class used to differentiate real values from complex ones."""

    __slots__ = ()


class FloatValue(RealValue):
    """Representation of a constant scalar floating point value."""

    __slots__ = ()

    def __getnewargs__(self):
        """Get new args."""
        return (self._value,)

    def __new__(cls, value):
        """Create a new FloatValue."""
        if value == 0.0:
            # Always represent zero with Zero
            return Zero()
        return ConstantValue.__new__(cls)

    def __init__(self, value):
        """Initialise."""
        super(FloatValue, self).__init__(float(value))

    def __repr__(self):
        """Representation."""
        r = "%s(%s)" % (type(self).__name__, format_float(self._value))
        return r


class IntValue(RealValue):
    """Representation of a constant scalar integer value."""

    __slots__ = ()

    _cache: typing.Dict[int, RealValue] = {}

    def __getnewargs__(self):
        """Get new args."""
        return (self._value,)

    def __new__(cls, value):
        """Create a new IntValue."""
        if value == 0:
            # Always represent zero with Zero
            return Zero()
        elif abs(value) < 100:
            # Small numbers are cached to reduce memory usage
            # (fly-weight pattern)
            self = IntValue._cache.get(value)
            if self is not None:
                return self
            self = RealValue.__new__(cls)
            IntValue._cache[value] = self
        else:
            self = RealValue.__new__(cls)
        self._init(value)
        return self

    def _init(self, value):
        """Initialise."""
        super(IntValue, self).__init__(int(value))

    def __init__(self, value):
        """Initialise."""
        super(IntValue, self).__init__(int(value))

    def __repr__(self):
        """Representation."""
        r = "%s(%s)" % (type(self).__name__, repr(self._value))
        return r

    def __hash__(self):
        """Hash."""
        return hash(f"{self!r}")


class Identity(ConstantValue):
    """Representation of an identity matrix."""

    __slots__ = "_dim"

    def __init__(self, dim):
        """Initialise."""
        ConstantValue.__init__(self, (dim, dim))
        self._dim = dim

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a, b = component
        return 1 if a == b else 0

    def __getitem__(self, key):
        """Get an item."""
        if len(key) != 2:
            raise ValueError("Size mismatch for Identity.")
        if all(isinstance(k, (int, FixedIndex)) for k in key):
            return IntValue(1) if (int(key[0]) == int(key[1])) else Zero()
        return Expr.__getitem__(self, key)

    def __str__(self):
        """Format as a string."""
        return "I"

    def __repr__(self):
        """Representation."""
        r = "Identity(%d)" % self._dim
        return r

    def __eq__(self, other):
        """Check equalty."""
        return isinstance(other, Identity) and self._dim == other._dim

    def __hash__(self):
        """Hash."""
        return super().__hash__()


class PermutationSymbol(ConstantValue):
    """Representation of a permutation symbol.

    This is also known as the Levi-Civita symbol, antisymmetric symbol,
    or alternating symbol.
    """

    __slots__ = "_dim"

    def __init__(self, dim):
        """Initialise."""
        ConstantValue.__init__(self, (dim,) * dim)
        self._dim = dim

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        return self.__eps(component)

    def __getitem__(self, key):
        """Get an item."""
        if len(key) != self._dim:
            raise ValueError("Size mismatch for PermutationSymbol.")
        if all(isinstance(k, (int, FixedIndex)) for k in key):
            return self.__eps(key)
        return Expr.__getitem__(self, key)

    def __str__(self):
        """Format as a string."""
        return "eps"

    def __repr__(self):
        """Representation."""
        r = "PermutationSymbol(%d)" % self._dim
        return r

    def __eq__(self, other):
        """Check equalty."""
        return isinstance(other, PermutationSymbol) and self._dim == other._dim

    def __hash__(self):
        """Hash."""
        return super().__hash__()

    def __eps(self, x):
        """Get eps.

        This function body is taken from
        http://www.mathkb.com/Uwe/Forum.aspx/math/29865/N-integer-Levi-Civita
        """
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
    """Converts expression to an Expr if possible."""
    if isinstance(expression, (Expr, ufl.BaseForm)):
        return expression
    elif isinstance(expression, complex):
        return ComplexValue(expression)
    elif isinstance(expression, float):
        return FloatValue(expression)
    elif isinstance(expression, int):
        return IntValue(expression)
    else:
        raise ValueError(
            f"Invalid type conversion: {expression} can not be converted to any UFL type."
        )
