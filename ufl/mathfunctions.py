"""This module provides basic mathematical functions."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008
# Modified by Kristian B. Oelgaard, 2011

import cmath
import math
import numbers
import warnings

from ufl.constantvalue import (ComplexValue, ConstantValue, FloatValue, IntValue, RealValue, Zero, as_ufl,
                               is_true_ufl_scalar)
from ufl.core.operator import Operator

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


class BesselJ(BesselFunction):
    """Bessel J function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_j", nu, argument)


class BesselY(BesselFunction):
    """Bessel Y function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_y", nu, argument)


class BesselI(BesselFunction):
    """Bessel I function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_i", nu, argument)


class BesselK(BesselFunction):
    """Bessel K function."""

    __slots__ = ()

    def __init__(self, nu, argument):
        """Initialise."""
        BesselFunction.__init__(self, "cyl_bessel_k", nu, argument)
