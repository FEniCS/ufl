"""This module provides basic mathematical functions."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008
# Modified by Kristian B. Oelgaard, 2011

import math
import cmath
import numbers
import warnings

from ufl.core.operator import Operator
from ufl.core.ufl_type import ufl_type
from ufl.constantvalue import (is_true_ufl_scalar, Zero, RealValue, FloatValue, IntValue, ComplexValue,
                               ConstantValue, as_ufl)

"""
TODO: Include additional functions available in <cmath> (need derivatives as well):

Exponential and logarithmic functions:
log10    Compute common logarithm (function)

TODO: Any other useful special functions?

About bessel functions:
http://en.wikipedia.org/wiki/Bessel_function

Portable implementations of bessel functions:
http://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/main_overview/tr1.html

Implementation in C++ std::tr1:: or boost::math::tr1::
- BesselK: cyl_bessel_k(nu, x)
- BesselI: cyl_bessel_i(nu, x)
- BesselJ: cyl_bessel_j(nu, x)
- BesselY: cyl_neumann(nu, x)
"""


# --- Function representations ---

@ufl_type(is_abstract=True, is_scalar=True, num_ops=1)
class MathFunction(Operator):
    """Base class for all unary scalar math functions."""

    # Freeze member variables for objects in this class
    __slots__ = ("_name",)

    def __init__(self, name, argument):
        """Initialise."""
        Operator.__init__(self, (argument,))
        if not is_true_ufl_scalar(argument):
            raise ValueError("Expecting scalar argument.")
        self._name = name

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        try:
            if isinstance(a, numbers.Real):
                res = getattr(math, self._name)(a)
            else:
                res = getattr(cmath, self._name)(a)
        except ValueError:
            warnings.warn('Value error in evaluation of function %s with argument %s.' % (self._name, a))
            raise
        return res

    def __str__(self):
        """Format as a string."""
        return "%s(%s)" % (self._name, self.ufl_operands[0])


@ufl_type()
class Sqrt(MathFunction):
    """Square root."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Sqrt."""
        if isinstance(argument, (RealValue, Zero, numbers.Real)):
            if float(argument) < 0:
                return ComplexValue(cmath.sqrt(complex(argument)))
            else:
                return FloatValue(math.sqrt(float(argument)))
        if isinstance(argument, (ComplexValue, complex)):
            return ComplexValue(cmath.sqrt(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "sqrt", argument)


@ufl_type()
class Exp(MathFunction):
    """Exponentiation.."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Exp."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.exp(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.exp(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "exp", argument)


@ufl_type()
class Ln(MathFunction):
    """Natural logarithm."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Ln."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.log(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.log(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "ln", argument)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        try:
            return math.log(a)
        except TypeError:
            return cmath.log(a)


@ufl_type()
class Cos(MathFunction):
    """Cosine."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Cos."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.cos(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.cos(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "cos", argument)


@ufl_type()
class Sin(MathFunction):
    """Sine."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Sin."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.sin(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.sin(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "sin", argument)


@ufl_type()
class Tan(MathFunction):
    """Tangent."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Tan."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.tan(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.tan(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "tan", argument)


@ufl_type()
class Cosh(MathFunction):
    """Hyperbolic cosine."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Cosh."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.cosh(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.cosh(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "cosh", argument)


@ufl_type()
class Sinh(MathFunction):
    """Hyperbolic sine."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Sinh."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.sinh(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.sinh(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "sinh", argument)


@ufl_type()
class Tanh(MathFunction):
    """Hyperbolic tangent."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Tanh."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.tanh(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.tanh(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "tanh", argument)


@ufl_type()
class Acos(MathFunction):
    """Inverse cosine."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Acos."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.acos(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.acos(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "acos", argument)


@ufl_type()
class Asin(MathFunction):
    """Inverse sine."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Asin."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.asin(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.asin(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "asin", argument)


@ufl_type()
class Atan(MathFunction):
    """Inverse tangent."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Atan."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.atan(float(argument)))
        if isinstance(argument, (ComplexValue)):
            return ComplexValue(cmath.atan(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "atan", argument)


@ufl_type(is_scalar=True, num_ops=2)
class Atan2(Operator):
    """Inverse tangent with two inputs."""

    __slots__ = ()

    def __new__(cls, arg1, arg2):
        """Create a new Atan2."""
        if isinstance(arg1, (RealValue, Zero)) and isinstance(arg2, (RealValue, Zero)):
            return FloatValue(math.atan2(float(arg1), float(arg2)))
        if isinstance(arg1, (ComplexValue)) or isinstance(arg2, (ComplexValue)):
            raise TypeError("Atan2 does not support complex numbers.")
        return Operator.__new__(cls)

    def __init__(self, arg1, arg2):
        """Initialise."""
        Operator.__init__(self, (arg1, arg2))
        if isinstance(arg1, (ComplexValue, complex)) or isinstance(arg2, (ComplexValue, complex)):
            raise TypeError("Atan2 does not support complex numbers.")
        if not is_true_ufl_scalar(arg1):
            raise ValueError("Expecting scalar argument 1.")
        if not is_true_ufl_scalar(arg2):
            raise ValueError("Expecting scalar argument 2.")

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        b = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        try:
            res = math.atan2(a, b)
        except TypeError:
            raise ValueError('Atan2 does not support complex numbers.')
        except ValueError:
            warnings.warn('Value error in evaluation of function atan2 with arguments %s, %s.' % (a, b))
            raise
        return res

    def __str__(self):
        """Format as a string."""
        return "atan2(%s,%s)" % (self.ufl_operands[0], self.ufl_operands[1])


@ufl_type()
class Erf(MathFunction):
    """Erf function."""

    __slots__ = ()

    def __new__(cls, argument):
        """Create a new Erf."""
        if isinstance(argument, (RealValue, Zero)):
            return FloatValue(math.erf(float(argument)))
        if isinstance(argument, (ConstantValue)):
            return ComplexValue(math.erf(complex(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        """Initialise."""
        MathFunction.__init__(self, "erf", argument)

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return math.erf(a)


@ufl_type(is_abstract=True, is_scalar=True, num_ops=2)
class BesselFunction(Operator):
    """Base class for all bessel functions."""

    __slots__ = ("_name")

    def __init__(self, name, nu, argument):
        """Initialise."""
        if not is_true_ufl_scalar(nu):
            raise ValueError("Expecting scalar nu.")
        if not is_true_ufl_scalar(argument):
            raise ValueError("Expecting scalar argument.")

        # Use integer representation if suitable
        fnu = float(nu)
        inu = int(nu)
        if fnu == inu:
            nu = as_ufl(inu)
        else:
            nu = as_ufl(fnu)

        Operator.__init__(self, (nu, argument))

        self._name = name

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        a = self.ufl_operands[1].evaluate(x, mapping, component, index_values)
        try:
            import scipy.special
        except ImportError:
            raise ValueError("You must have scipy installed to evaluate bessel functions in python.")
        name = self._name[-1]
        if isinstance(self.ufl_operands[0], IntValue):
            nu = int(self.ufl_operands[0])
            functype = 'n' if name != 'i' else 'v'
        else:
            nu = self.ufl_operands[0].evaluate(x, mapping, component,
                                               index_values)
            functype = 'v'
        func = getattr(scipy.special, name + functype)
        return func(nu, a)

    def __str__(self):
        """Format as a string."""
        return "%s(%s, %s)" % (self._name, self.ufl_operands[0],
                               self.ufl_operands[1])


@ufl_type()
class BesselJ(BesselFunction):
    """Bessel J function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_j", nu, argument)


@ufl_type()
class BesselY(BesselFunction):
    """Bessel Y function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_y", nu, argument)


@ufl_type()
class BesselI(BesselFunction):
    """Bessel I function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_i", nu, argument)


@ufl_type()
class BesselK(BesselFunction):
    """Bessel K function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_k", nu, argument)
