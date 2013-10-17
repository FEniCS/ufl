"""This module defines partial differentiation rules for
all relevant operands for use with reverse mode AD."""

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
# Modified by Kristian B. Oelgaard, 2011
#
# First added:  2009-01-06
# Last changed: 2012-04-12

from math import pi
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.classes import Zero, IntValue
from ufl.operators import cos, sin, cosh, sinh, exp, ln, sqrt, conditional, sign
from ufl.tensors import unit_vectors, ListTensor
from ufl.algorithms.multifunction import MultiFunction

class PartialDerivativeComputer(MultiFunction):
    """NB! The main reason for keeping this out of the Expr hierarchy is
    to avoid user mistakes in the form of mixups with total derivatives,
    and to allow both reverse and forward mode AD."""
    #def __init__(self, spatial_dim):
        #self._spatial_dim = spatial_dim
    def __init__(self):
        MultiFunction.__init__(self)

    # TODO: Make sure we have implemented partial derivatives of all operators.
    #        At least non-compound ones should be covered, but compound ones
    #        may be a good idea in future versions.

    def expr(self, o):
        error("No partial derivative defined for %s" % type(o))

    # --- Basic algebra operators

    def index_sum(self, f):
        "d/dx sum_j x = TODO"
        TODO

    def sum(self, f):
        "d/dx_i sum_j x_j = 1"
        #_1 = IntValue(1, o.free_indices(), o.index_dimensions())
        _1 = IntValue(1) # TODO: Handle non-scalars
        return (_1,)*len(f.operands())

    def product(self, f):
        a, b = f.operands() # TODO: Assuming binary operator for now
        da = b # TODO: Is this right even for non-scalar b?
        db = a
        return (da, db)

    def division(self, f):
        """f = x/y
        d/dx x/y = 1/y
        d/dy x/y = -x/y**2 = -f/y"""
        x, y = f.operands()
        # Nonscalar x not supported
        ufl_assert(x.shape() == (), "Expecting scalars in division.")
        ufl_assert(y.shape() == (), "Expecting scalars in division.")
        d = 1 / y
        return (d, -f*d)

    def power(self, f):
        """f = x**y
        d/dx x**y = y*x**(y-1) = y*f/x
        d/dy x**y = ln(x)*x**y = ln(x)*f"""
        x, y = f.operands()
        dx = y*f/x
        dy = ln(x)*f
        return (dx, dy)

    def abs(self, f):
        r".. math:: \\frac{d}{dx} f(x) = \\frac{d}{dx} abs(x) = sign(x)"
        x, = f.operands()
        dx = sign(x)
        return (dx,)

    # --- Mathfunctions

    def sqrt(self, f):
        "d/dx sqrt(x) = 1 / (2*sqrt(x))"
        return (0.5/f,)

    def exp(self, f):
        "d/dx exp(x) = exp(x)"
        return (f,)

    def ln(self, f):
        "d/dx ln x = 1 / x"
        x, = f.operands()
        return (1/x,)

    def cos(self, f):
        "d/dx cos x = -sin(x)"
        x, = f.operands()
        return (-sin(x),)

    def sin(self, f):
        "d/dx sin x = cos(x)"
        x, = f.operands()
        return (cos(x),)

    def tan(self, f):
        "d/dx tan x = (sec(x))^2 = 2/(cos(2x) + 1)"
        x, = f.operands()
        return (2.0/(cos(2.0*x) + 1.0),)

    def cosh(self, f):
        "d/dx cosh x = sinh(x)"
        x, = f.operands()
        return (sinh(x),)

    def sinh(self, f):
        "d/dx sinh x = cosh(x)"
        x, = f.operands()
        return (cosh(x),)

    def tanh(self, f):
        "d/dx tanh x = (sech(x))^2 = (2 cosh(x) / (cosh(2x) + 1))^2"
        x, = f.operands()
        return (((2.0*cosh(x))/(cosh(2.0*x) + 1.0))**2,)

    def acos(self, f):
        r".. math:: \\frac{d}{dx} f(x) = \frac{d}{dx} \arccos(x) = \frac{-1}{\sqrt{1 - x^2}}"
        x, = f.operands()
        return (-1.0/sqrt(1.0 - x**2),)

    def asin(self, f):
        "d/dx asin x = 1/sqrt(1 - x^2)"
        x, = f.operands()
        return (1.0/sqrt(1.0 - x**2),)

    def atan(self, f):
        "d/dx atan x = 1/(1 + x^2)"
        x, = f.operands()
        return (1.0/(1.0 + x**2),)

    def atan_2(self, f):
        """
        f = atan2(x,y)
        d/dx atan2(x,y) = y / (x**2 + y**2 ) 
        d/dy atan2(x,y) = -x / (x**2 + y**2)
        """
        x,y = f.operands()
        d = x**2 + y**2
        return (y/d,-x/d)

    def erf(self, f):
        "d/dx erf x = 2/sqrt(pi)*exp(-x^2)"
        x, = f.operands()
        return (2.0/sqrt(pi)*exp(-x**2),)

    def bessel_function(self, nu, x):
        return NotImplemented

    # --- Shape and indexing manipulators

    def indexed(self, f): # TODO: Is this right? Fix for non-scalars too.
        "d/dx x_i = (1)_i = 1"
        s = f.shape()
        ufl_assert(s == (), "TODO: Assuming a scalar expression.")
        _1 = IntValue(1) # TODO: Non-scalars
        return (_1, None)

    def list_tensor(self, f): # TODO: Is this right? Fix for higher order tensors too.
        "d/dx_i [x_0, ..., x_n-1] = e_i (unit vector)"
        ops = f.operands()
        n = len(ops)
        s = ops[0].shape()
        ufl_assert(s == (), "TODO: Assuming a vector, i.e. scalar operands.")
        return unit_vectors(n) # TODO: Non-scalars

    def component_tensor(self, f):
        x, i = f.operands()
        s = f.shape()
        ufl_assert(len(s) == 1, "TODO: Assuming a vector, i.e. scalar operands.")
        n, = s
        d = ListTensor([1]*n) # TODO: Non-scalars
        return (d, None)

    # --- Restrictions

    def positive_restricted(self, f):
        _1 = IntValue(1)
        return (_1,) # or _1('+')? TODO: is this right?
        # Note that _1('+') would become 0 with the current implementation

    def negative_restricted(self, f):
        _1 = IntValue(1)
        return (_1,) # or _1('-')? TODO: is this right?

    def cell_avg(self, f):
        error("Not sure how to implement partial derivative of this operator at all actually.")

    def facet_avg(self, f):
        error("Not sure how to implement partial derivative of this operator at all actually.")

    # --- Conditionals

    def condition(self, f):
        return (None, None)

    def conditional(self, f): # TODO: Is this right? What about non-scalars?
        c, a, b = f.operands()
        s = f.shape()
        ufl_assert(s == (), "TODO: Assuming scalar valued expressions.")
        _0 = Zero()
        _1 = IntValue(1)
        da = conditional(c, _1, _0)
        db = conditional(c, _0, _1)
        return (None, da, db)

    # --- Derivatives

    def spatial_derivative(self, f):
        error("Partial derivative of spatial_derivative not implemented, "\
              "when is this called? apply_ad should make sure it isn't called.")
        x, i = f.operands()
        return (None, None)

    def variable_derivative(self, f):
        error("Partial derivative of variable_derivative not implemented, "\
              "when is this called? apply_ad should make sure it isn't called.")
        x, v = f.operands()
        return (None, None)

    def coefficient_derivative(self, f):
        error("Partial derivative of coefficient_derivative not implemented, "\
              "when is this called? apply_ad should make sure it isn't called.")
        a, w, v = f.operands()
        return (None, None, None)

# Example usage:
def pdiffs(exprs):
    pd = PartialDerivativeComputer()
    return [pd(e) for e in exprs]

