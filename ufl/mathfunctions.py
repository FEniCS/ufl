"""This module provides basic mathematical functions."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# Last changed: 2011-10-25

import math
from ufl.log import warning, error
from ufl.assertions import ufl_assert
from ufl.expr import Operator
from ufl.constantvalue import is_true_ufl_scalar, ScalarValue, Zero, FloatValue, IntValue

"""
TODO: Include additional functions available in <cmath> (need derivatives as well):

Trigonometric functions:
atan2    Compute arc tangent with two parameters (function)

Hyperbolic functions:
cosh     Compute hyperbolic cosine (function)
sinh     Compute hyperbolic sine (function)
tanh     Compute hyperbolic tangent (function)

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
        return {}
    
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
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.sqrt(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "sqrt", argument)

class Exp(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.exp(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "exp", argument)

class Ln(MathFunction):
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
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.cos(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "cos", argument)

class Sin(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.sin(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "sin", argument)

class Tan(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.tan(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "tan", argument)

class Acos(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.acos(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "acos", argument)

class Asin(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.asin(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "asin", argument)

class Atan(MathFunction):
    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.atan(float(argument)))
        return MathFunction.__new__(cls)
    
    def __init__(self, argument):
        MathFunction.__init__(self, "atan", argument)

def _find_erf():
    import math
    if hasattr(math, 'erf'):
        return math.erf
    import scipy.special
    if hasattr(scipy.special, 'erf'):
        return scipy.special.erf
    return None

class Erf(MathFunction):
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
        ufl_assert(is_true_ufl_scalar(nu), "Expecting scalar argument.")
        ufl_assert(is_true_ufl_scalar(argument), "Expecting scalar argument.")
        self._classname = classname
        self._name     = name
        self._nu       = nu
        self._argument = argument

    def operands(self):
        return (self._nu, self._argument)

    def free_indices(self):
        return ()

    def index_dimensions(self):
        return {}

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
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_j", "BesselJ", nu, argument)

class BesselY(BesselFunction):
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_y", "BesselY", nu, argument)

class BesselI(BesselFunction):
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_i", "BesselI", nu, argument)

class BesselK(BesselFunction):
    def __init__(self, nu, argument):
        BesselFunction.__init__(self, "cyl_bessel_k", "BesselK", nu, argument)
