"""Algorithms for estimating polynomial degrees of expressions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010
# Modified by Jan Blechta, 2012

import warnings

from ufl.corealg.multifunction import MultiFunction
from ufl.checks import is_cellwise_constant
from ufl.constantvalue import IntValue
from ufl.corealg.map_dag import map_expr_dags
from ufl.domain import extract_unique_domain
from ufl.form import Form
from ufl.integral import Integral


class SumDegreeEstimator(MultiFunction):
    """Sum degree estimator.

    This algorithm is exact for a few operators and heuristic for many.
    """

    def __init__(self, default_degree, element_replace_map):
        """Initialise."""
        MultiFunction.__init__(self)
        self.default_degree = default_degree
        self.element_replace_map = element_replace_map

    def constant_value(self, v):
        """Apply to constant_value.

        Constant values are constant.
        """
        return 0

    def constant(self, v):
        """Apply to constant."""
        return 0

    def geometric_quantity(self, v):
        """Apply to geometric_quantity.

        Some geometric quantities are cellwise constant. Others are nonpolynomial and thus hard to estimate.
        """
        if is_cellwise_constant(v):
            return 0
        else:
            # As a heuristic, just returning domain degree to bump up degree somewhat
            return extract_unique_domain(v).ufl_coordinate_element().degree()

    def spatial_coordinate(self, v):
        """Apply to spatial_coordinate.

        A coordinate provides additional degrees depending on coordinate field of domain.
        """
        return extract_unique_domain(v).ufl_coordinate_element().degree()

    def cell_coordinate(self, v):
        """Apply to cell_coordinate.

        A coordinate provides one additional degree.
        """
        return 1

    def argument(self, v):
        """Apply to argument.

        A form argument provides a degree depending on the element,
        or the default degree if the element has no degree.
        """
        return v.ufl_element().degree()  # FIXME: Use component to improve accuracy for mixed elements

    def coefficient(self, v):
        """Apply to coefficient.

        A form argument provides a degree depending on the element,
        or the default degree if the element has no degree.
        """
        e = v.ufl_element()
        e = self.element_replace_map.get(e, e)
        d = e.degree()  # FIXME: Use component to improve accuracy for mixed elements
        if d is None:
            d = self.default_degree
        return d

    def _reduce_degree(self, v, f):
        """Apply to _reduce_degree.

        Reduces the estimated degree by one; used when derivatives
        are taken. Does not reduce the degree when TensorProduct elements
        or quadrilateral elements are involved.
        """
        if isinstance(f, int) and v.ufl_domain().ufl_cell().cellname() not in ["quadrilateral", "hexahedron"]:
            return max(f - 1, 0)
        else:
            return f

    def _add_degrees(self, v, *ops):
        """Apply to _add_degrees."""
        if any(isinstance(o, tuple) for o in ops):
            # we can add a slight hack here to handle things
            # like adding 0 to (3, 3) [by expanding
            # 0 to (0, 0) when making tempops]
            tempops = [foo if isinstance(foo, tuple) else (foo, foo) for foo in ops]
            return tuple(map(sum, zip(*tempops)))
        else:
            return sum(ops)

    def _max_degrees(self, v, *ops):
        """Apply to _max_degrees."""
        if any(isinstance(o, tuple) for o in ops):
            tempops = [foo if isinstance(foo, tuple) else (foo, foo) for foo in ops]
            return tuple(map(max, zip(*tempops)))
        else:
            return max(ops + (0,))

    def _not_handled(self, v, *args):
        """Apply to _not_handled."""
        raise ValueError(f"Missing degree handler for type {v._ufl_class_.__name__}")

    def expr(self, v, *ops):
        """Apply to expr.

        For most operators we take the max degree of its operands.
        """
        warnings.warn(f"Missing degree estimation handler for type {v._ufl_class_.__name__}")
        return self._add_degrees(v, *ops)

    # Utility types with no degree concept
    def multi_index(self, v):
        """Apply to multi_index."""
        return None

    def label(self, v):
        """Apply to label."""
        return None

    # Fall-through, indexing and similar types
    def reference_value(self, rv, f):
        """Apply to reference_value."""
        return f

    def variable(self, v, e, a):
        """Apply to variable."""
        return e

    def transposed(self, v, A):
        """Apply to transposed."""
        return A

    def index_sum(self, v, A, ii):
        """Apply to index_sum."""
        return A

    def indexed(self, v, A, ii):
        """Apply to indexed."""
        return A

    def component_tensor(self, v, A, ii):
        """Apply to component_tensor."""
        return A

    list_tensor = _max_degrees

    def positive_restricted(self, v, a):
        """Apply to positive_restricted."""
        return a

    def negative_restricted(self, v, a):
        """Apply to negative_restricted."""
        return a

    def conj(self, v, a):
        """Apply to conj."""
        return a

    def real(self, v, a):
        """Apply to real."""
        return a

    def imag(self, v, a):
        """Apply to imag."""
        return a

    # A sum takes the max degree of its operands:
    sum = _max_degrees

    # TODO: Need a new algorithm which considers direction of
    # derivatives of form arguments A spatial derivative reduces the
    # degree with one
    grad = _reduce_degree
    reference_grad = _reduce_degree
    # Handling these types although they should not occur... please
    # apply preprocessing before using this algorithm:
    nabla_grad = _reduce_degree
    div = _reduce_degree
    reference_div = _reduce_degree
    nabla_div = _reduce_degree
    curl = _reduce_degree
    reference_curl = _reduce_degree

    def cell_avg(self, v, a):
        """Apply to cell_avg.

        Cell average of a function is always cellwise constant.
        """
        return 0

    def facet_avg(self, v, a):
        """Apply to facet_avg.

        Facet average of a function is always cellwise constant.
        """
        return 0

    # A product accumulates the degrees of its operands:
    product = _add_degrees
    # Handling these types although they should not occur... please
    # apply preprocessing before using this algorithm:
    inner = _add_degrees
    dot = _add_degrees
    outer = _add_degrees
    cross = _add_degrees

    # Explicitly not handling these types, please apply preprocessing
    # before using this algorithm:
    derivative = _not_handled  # base type
    compound_derivative = _not_handled  # base type
    compound_tensor_operator = _not_handled  # base class
    variable_derivative = _not_handled
    trace = _not_handled
    determinant = _not_handled
    cofactor = _not_handled
    inverse = _not_handled
    deviatoric = _not_handled
    skew = _not_handled
    sym = _not_handled

    def abs(self, v, a):
        """Apply to abs.

        This is a heuristic, correct if there is no.
        """
        if a == 0:
            return a
        else:
            return a

    def division(self, v, *ops):
        """Apply to division.

        Using the sum here is a heuristic. Consider e.g. (x+1)/(x-1).
        """
        return self._add_degrees(v, *ops)

    def power(self, v, a, b):
        """Apply to power.

        If b is a positive integer: degree(a**b) == degree(a)*b
        otherwise use the heuristic: degree(a**b) == degree(a) + 2.
        """
        f, g = v.ufl_operands

        if isinstance(g, IntValue):
            gi = g.value()
            if gi >= 0:
                if isinstance(a, int):
                    return a * gi
                else:
                    return tuple(foo * gi for foo in a)

        # Something to a non-(positive integer) power, e.g. float,
        # negative integer, Coefficient, etc.
        return self._add_degrees(v, a, 2)

    def atan2(self, v, a, b):
        """Apply to atan2.

        Using the heuristic:
        degree(atan2(const,const)) == 0
        degree(atan2(a,b)) == max(degree(a),degree(b))+2
        which can be wildly inaccurate but at least gives a somewhat high integration degree.
        """
        if a or b:
            return self._add_degrees(v, self._max_degrees(v, a, b), 2)
        else:
            return self._max_degrees(v, a, b)

    def math_function(self, v, a):
        """Apply to math_function.

        Using the heuristic:
        degree(sin(const)) == 0
        degree(sin(a)) == degree(a)+2
        which can be wildly inaccurate but at least gives a somewhat high integration degree.
        """
        if a:
            return self._add_degrees(v, a, 2)
        else:
            return a

    def bessel_function(self, v, nu, x):
        """Apply to bessel_function.

        Using the heuristic
        degree(bessel_*(const)) == 0
        degree(bessel_*(x)) == degree(x)+2
        which can be wildly inaccurate but at least gives a somewhat high integration degree.
        """
        if x:
            return self._add_degrees(v, x, 2)
        else:
            return x

    def condition(self, v, *args):
        """Apply to condition."""
        return None

    def conditional(self, v, c, t, f):
        """Apply to conditional.

        Degree of condition does not influence degree of values which conditional takes. So
        heuristicaly taking max of true degree and false degree. This will be exact in cells
        where condition takes single value. For improving accuracy of quadrature near
        condition transition surface quadrature order must be adjusted manually.
        """
        return self._max_degrees(v, t, f)

    def min_value(self, v, a, r):
        """Apply to min_value.

        Same as conditional.
        """
        return self._max_degrees(v, a, r)
    max_value = min_value

    def coordinate_derivative(self, v, integrand_degree, b, direction_degree, d):
        """Apply to coordinate_derivative.

        We use the heuristic that a shape derivative in direction V
        introduces terms V and grad(V) into the integrand. Hence we add the
        degree of the deformation to the estimate.
        """
        return self._add_degrees(v, integrand_degree, direction_degree)

    def expr_list(self, v, *o):
        """Apply to expr_list."""
        return self._max_degrees(v, *o)

    def expr_mapping(self, v, *o):
        """Apply to expr_mapping."""
        return self._max_degrees(v, *o)


def estimate_total_polynomial_degree(e, default_degree=1,
                                     element_replace_map={}):
    """Estimate total polynomial degree of integrand.

    NB: Although some compound types are supported here,
    some derivatives and compounds must be preprocessed
    prior to degree estimation. In generic code, this algorithm
    should only be applied after preprocessing.

    For coefficients defined on an element with unspecified degree (None),
    the degree is set to the given default degree.
    """
    de = SumDegreeEstimator(default_degree, element_replace_map)
    if isinstance(e, Form):
        if not e.integrals():
            raise ValueError("Form has no integrals.")
        degrees = map_expr_dags(de, [it.integrand() for it in e.integrals()])
    elif isinstance(e, Integral):
        degrees = map_expr_dags(de, [e.integrand()])
    else:
        degrees = map_expr_dags(de, [e])
    degree = max(degrees) if degrees else default_degree
    return degree
