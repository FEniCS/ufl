"""This module provides basic mathematical functions."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Modified by Anders Logg, 2008
# Modified by Kristian B. Oelgaard, 2011
#
# First added:  2008-03-14
# Last changed: 2013-03-15

import math
from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.operatorbase import Operator
from ufl.constantvalue import is_true_ufl_scalar, ScalarValue, Zero, FloatValue, IntValue, as_ufl
from ufl.common import EmptyDict

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

#--- Function representations ---

class MathFunction(Operator):
    "Base class for all math functions"
    # Freeze member variables for objects in this class
    __slots__ = ("_name", "_argument",)
    def __init__(self, name, argument):
        Operator.__init__(self)
        ufl_assert(is_true_ufl_scalar(argument), "Expecting scalar argument.")
        self._name     = name
        self._argument = argument

    def operands(self):
        return (self._argument,)

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return EmptyDict

    def shape(self):
        return ()

    def evaluate(self, x, mapping, component, index_values):
        a = self._argument.evaluate(x, mapping, component, index_values)
        try:
            res = getattr(math, self._name)(a)
        except ValueError:
            warning('Value error in evaluation of function %s with argument %s.' % (self._name, a))
            raise
        return res

    def __str__(self):
        return "%s(%s)" % (self._name, self._argument)

    def __repr__(self):
        return "%s(%r)" % (self._name, self._argument)

class Sqrt(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.sqrt(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "sqrt", argument)

class Exp(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.exp(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "exp", argument)

class Ln(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.log(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "ln", argument)

    def evaluate(self, x, mapping, component, index_values):
        a = self._argument.evaluate(x, mapping, component, index_values)
        return math.log(a)

class Cos(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.cos(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "cos", argument)

class Sin(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.sin(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "sin", argument)

class Tan(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.tan(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "tan", argument)

class Cosh(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.cosh(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "cosh", argument)

class Sinh(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.sinh(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "sinh", argument)

class Tanh(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.tanh(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "tanh", argument)

class Acos(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.acos(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "acos", argument)

class Asin(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.asin(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "asin", argument)

class Atan(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.atan(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "atan", argument)

class Atan2(Operator):
    __slots__ = ("_name", "_arg1", "_arg2", "_classname")
    def __new__(cls, arg1, arg2):
        if isinstance(arg1, (ScalarValue, Zero)) and isinstance(arg2, (ScalarValue, Zero)):
            return FloatValue(math.atan2(float(arg1), float(arg2)))
        return Operator.__new__(cls)

    def __init__(self, arg1, arg2):
        Operator.__init__(self)
        ufl_assert(is_true_ufl_scalar(arg1), "Expecting scalar argument 1.")
        ufl_assert(is_true_ufl_scalar(arg2), "Expecting scalar argument 2.")
        self._name     = "atan_2"
        self._arg1 = arg1
        self._arg2 = arg2

    def operands(self):
        return (self._arg1, self._arg2)

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return EmptyDict

    def shape(self):
        return ()

    def evaluate(self, x, mapping, component, index_values):
        a = self._arg1.evaluate(x, mapping, component, index_values)
        b = self._arg2.evaluate(x, mapping, component, index_values)
        try:
            res = math.atan2(a,b)
        except ValueError:
            warning('Value error in evaluation of function %s with arguments %s, %s.' % (self._name, a,b))
            raise
        return res

    def __str__(self):
        return "%s(%s,%s)" % (self._name, self._arg1, self._arg2)

    def __repr__(self):
        return "%s(%s,%s)" % (self._name, self._arg1, self._arg2)


def _find_erf():
    import math
    if hasattr(math, 'erf'):
        return math.erf
    import scipy.special
    if hasattr(scipy.special, 'erf'):
        return scipy.special.erf
    return None

class Erf(MathFunction):
    __slots__ = ()
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            erf = _find_erf()
            if erf is not None:
                return FloatValue(erf(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "erf", argument)

    def evaluate(self, x, mapping, component, index_values):
        a = self._argument.evaluate(x, mapping, component, index_values)
        erf = _find_erf()
        if erf is None:
            error("No python implementation of erf available on this system, cannot evaluate. Upgrade python or install scipy.")
        return erf(a)

class BesselFunction(Operator):
    "Base class for all bessel functions"
    # Freeze member variables for objects in this class
    __slots__ = ("_name", "_nu", "_argument", "_classname")
    def __init__(self, name, classname, nu, argument):
        Operator.__init__(self)
        ufl_assert(is_true_ufl_scalar(nu), "Expecting scalar nu.")
        ufl_assert(is_true_ufl_scalar(argument), "Expecting scalar argument.")
        fnu = float(nu)
        inu = int(nu)
        if fnu == inu:
            nu = as_ufl(inu)
        else:
            nu = as_ufl(fnu)
        self._classname = classname
        self._name     = name
        self._nu       = nu
        self._argument = argument

    def operands(self):
        return (self._nu, self._argument)

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return EmptyDict

    def shape(self):
        return ()

    def evaluate(self, x, mapping, component, index_values):
        a = self._argument.evaluate(x, mapping, component, index_values)
        try:
            import scipy.special
        except:
            error("You must have scipy installed to evaluate bessel functions in python.")
        name = self._name[-1]
        if isinstance(self._nu, IntValue):
            nu = int(self._nu)
            functype = 'n' if name != 'i' else 'v'
        else:
            nu = self._nu.evaluate(x, mapping, component, index_values)
            functype = 'v'
        func = getattr(scipy.special, name + functype)
        return func(nu, a)

    def __str__(self):
        return "%s(%s, %s)" % (self._name, self._nu, self._argument)

    def __repr__(self):
        return "%s(%r, %r)" % (self._classname, self._nu, self._argument)

class BesselJ(BesselFunction):
    __slots__ = ()
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_j", "BesselJ", nu, argument)

class BesselY(BesselFunction):
    __slots__ = ()
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_y", "BesselY", nu, argument)

class BesselI(BesselFunction):
    __slots__ = ()
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_i", "BesselI", nu, argument)

class BesselK(BesselFunction):
    __slots__ = ()
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_k", "BesselK", nu, argument)
