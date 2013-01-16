"""Algorithms for estimating polynomial degrees of expressions."""

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
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
# Modified by Anders Logg, 2009-2010
# Modified by Jan Blechta, 2012
#
# First added:  2008-05-07
# Last changed: 2012-11-04

from ufl.assertions import ufl_assert
from ufl.log import warning
from ufl.form import Form
from ufl.integral import Integral
from ufl.algorithms.transformer import Transformer


class SumDegreeEstimator(Transformer):
    "This algorithm is exact for a few operators and heuristic for many."

    def __init__(self, default_degree):
        Transformer.__init__(self)
        self.default_degree = default_degree

    def terminal(self, v):
        "Most terminals are spatially constant."
        return 0

    def spatial_coordinate(self, v):
        "A coordinate provides one additional degree."
        return 1

    def form_argument(self, v):
        """A form argument provides a degree depending on the element,
        or the default degree if the element has no degree."""
        d = v.element().degree() # FIXME: Use component to improve accuracy
        return self.default_degree if d is None else d

    def expr(self, v, *ops):
        "For most operators we take the max degree of its operands."
        return max(ops)

    def spatial_derivative(self, v, f, i):
        "A spatial derivative reduces the degree with one."
        return max(f - 1, 0)

    def grad(self, v, f):
        "A spatial derivative reduces the degree with one."
        return max(f - 1, 0)

    def product(self, v, *ops):
        "Using the sum here is exact."
        return sum(ops)

    def division(self, v, *ops):
        "Using the sum here is a heuristic. Consider e.g. (x+1)/(x-1)."
        return sum(ops)

    def power(self, v, a, b):
        """If b is an integer:
        degree(a**b) == degree(a)*b
        otherwise use the heuristic
        degree(a**b) == degree(a)*2"""
        f, g = v.operands()
        try:
            gi = int(g)
            return a*gi
        except:
            pass
        # Something to a non-integer power, this is just a heuristic with no background
        return a*2

    def math_function(self, v, a):
        """Using the heuristic
        degree(sin(const)) == 0
        degree(sin(a)) == degree(a)+2
        which can be wildly inaccurate but at least
        gives a somewhat high integration degree.
        """
        if a:
            return a+2
        else:
            return a

    def bessel_function(self, v, nu, x):
        """Using the heuristic
        degree(bessel_*(const)) == 0
        degree(bessel_*(x)) == degree(x)+2
        which can be wildly inaccurate but at least
        gives a somewhat high integration degree.
        """
        if x:
            return x+2
        else:
            return x

    def conditional(self, v, c, t, f):
        """Degree of condition does not
        influence degree of values which
        conditional takes. So heuristicaly
        taking max of true degree and false
        degree. This will be exact in cells
        where condition takes single value.
        For improving accuracy of quadrature
        near condition transition surface
        quadrature order must be adjusted manually."""
        return max(t, f)

class MaxDegreeEstimator(Transformer):
    def __init__(self, default_degree):
        Transformer.__init__(self)
        self.default_degree = default_degree

    def terminal(self, v):
        return 0

    def expr(self, v, *ops):
        return max(ops)

    def form_argument(self, v):
        return v.element().degree() # FIXME: Use component to improve accuracy

    #def spatial_derivative(self, v, f, i):
    #    return max(f - 1, 0)
    #def grad(self, v, f):
    #    return max(f - 1, 0)

    def product(self, v, *ops):
        degrees = [op for op in ops if not op is None]
        nones = [op for op in ops if op is None]
        return max(degrees + [self.default_degree])

def estimate_max_polynomial_degree(e, default_degree=1):
    """Estimate the maximum polymomial degree of all functions in the
    expression. For coefficients defined on an element with unspecified
    degree (None), the degree is set to the given default degree."""
    de = MaxDegreeEstimator(default_degree)
    if isinstance(e, Form):
        ufl_assert(e.integrals(), "Got form with no integrals!")
        degrees = [de.visit(integral.integrand()) for integral in e.integrals()]
    elif isinstance(e, Integral):
        degrees = [de.visit(e.integrand())]
    else:
        degrees = [de.visit(e)]
    return max(degrees + [0])

def estimate_total_polynomial_degree(e, default_degree=1):
    """Estimate total polynomial degree of integrand. For coefficients
    defined on an element with unspecified degree (None), the degree
    is set to the given default degree."""
    de = SumDegreeEstimator(default_degree)
    if isinstance(e, Form):
        ufl_assert(e.integrals(), "Got form with no integrals!")
        degrees = [de.visit(integral.integrand()) for integral in e.integrals()]
    elif isinstance(e, Integral):
        degrees = [de.visit(e.integrand())]
    else:
        degrees = [de.visit(e)]
    return max(degrees + [0])
