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

    def __init__(self, default_degree, element_replace_map):
        Transformer.__init__(self)
        self.default_degree = default_degree
        self.element_replace_map = element_replace_map

    def constant_value(self, v):
        "Constant values are constant. Duh."
        return 0

    def geometric_quantity(self, v):
        "Most geometric quantities are cellwise constant."
        ufl_assert(v.is_cellwise_constant(),
                   "Missing handler for non-constant geometry type %s." % v._uflclass.__name__)
        return 0

    def spatial_coordinate(self, v):
        "A coordinate provides one additional degree."
        return 1

    def local_coordinate(self, v):
        "A coordinate provides one additional degree."
        return 1

    def form_argument(self, v):
        """A form argument provides a degree depending on the element,
        or the default degree if the element has no degree."""
        e = v.element()
        e = self.element_replace_map.get(e,e)
        d = e.degree() # FIXME: Use component to improve accuracy
        return self.default_degree if d is None else d

    def _reduce_degree(self, v, f):
        return max(f - 1, 0)
    def _add_degrees(self, v, *ops):
        return sum(ops)
    def _max_degrees(self, v, *ops):
        return max(ops + (0,))
    def _not_handled(self, v, *args):
        error("Missing degree handler for type %s" % v._uflclass.__name__)

    def expr(self, v, *ops):
        "For most operators we take the max degree of its operands."
        warning("Missing degree estimation handler for type %s" % v._uflclass.__name__)
        return self._add_degrees(v, *ops)

    # Utility types with no degree concept
    def multi_index(self, v):
        return None
    def label(self, v):
        return None
    # Fall-through, indexing and similar types
    def variable(self, v, e, l):
        return e
    def transposed(self, v, A):
        return A
    def index_sum(self, v, A, ii):
        return A
    def indexed(self, v, A, ii):
        return A
    def component_tensor(self, v, A, ii):
        return A
    list_tensor = _max_degrees
    def positive_restricted(self, v, a):
        return a
    def negative_restricted(self, v, a):
        return a

    # A sum takes the max degree of its operands:
    sum = _max_degrees

    # A spatial derivative reduces the degree with one
    grad = _reduce_degree
    # Handling these types although they should not occur... please apply preprocessing before using this algorithm:
    nabla_grad = _reduce_degree
    div = _reduce_degree
    nabla_div = _reduce_degree
    curl = _reduce_degree

    def cell_avg(self, v, a):
        "Cell average of a function is always cellwise constant."
        return 0

    def facet_avg(self, v, a):
        "Facet average of a function is always cellwise constant."
        return 0

    # A product accumulates the degrees of its operands:
    product = _add_degrees
    # Handling these types although they should not occur... please apply preprocessing before using this algorithm:
    inner = _add_degrees
    dot = _add_degrees
    outer = _add_degrees
    cross = _add_degrees

    # Explicitly not handling these types, please apply preprocessing before using this algorithm:
    derivative = _not_handled # base type
    compound_derivative = _not_handled # base type
    compound_tensor_operator = _not_handled # base class
    variable_derivative = _not_handled
    trace = _not_handled
    determinant = _not_handled
    cofactor = _not_handled
    inverse = _not_handled
    deviatoric = _not_handled
    skew = _not_handled
    sym = _not_handled

    def abs(self, v, a):
        "This is a heuristic, correct if there is no "
        if a == 0:
            return a
        else:
            return a

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
            gi = abs(int(g))
            return a*gi
        except:
            pass
        # Something to a non-integer power, this is just a heuristic with no background
        return a*2

    def atan_2(self, v, a, b):
        """Using the heuristic
        degree(atan2(const,const)) == 0
        degree(atan2(a,b)) == max(degree(a),degree(b))+2
        which can be wildly inaccurate but at least
        gives a somewhat high integration degree.
        """
        #print "estimate",a,b
        if a or b:
            return max(a,b)+2
        else:
            return max(a,b)


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

    def condition(self, v, *args):
        return None

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

def estimate_total_polynomial_degree(e, default_degree=1, element_replace_map={}):
    """Estimate total polynomial degree of integrand.

    NB! Although some compound types are supported here,
    some derivatives and compounds must be preprocessed
    prior to degree estimation. In generic code, this algorithm
    should only be applied after preprocessing.

    For coefficients defined on an element with unspecified degree (None),
    the degree is set to the given default degree.
    """
    de = SumDegreeEstimator(default_degree, element_replace_map)
    if isinstance(e, Form):
        ufl_assert(e.integrals(), "Got form with no integrals!")
        degrees = [de.visit(integral.integrand()) for integral in e.integrals()]
    elif isinstance(e, Integral):
        degrees = [de.visit(e.integrand())]
    else:
        degrees = [de.visit(e)]
    return max(degrees + [0])
