# -*- coding: utf-8 -*-
"""Algorithms for estimating polynomial degrees of expressions."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010
# Modified by Jan Blechta, 2012

from ufl.log import warning, error
from ufl.form import Form
from ufl.integral import Integral
from ufl.algorithms.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dags
from ufl.checks import is_cellwise_constant
from ufl.constantvalue import IntValue


class IrreducibleInt(int):
    """Degree type used by quadrilaterals.

    Unlike int, values of this type are not decremeneted by _reduce_degree.
    """
    pass


class SumDegreeEstimator(MultiFunction):
    "This algorithm is exact for a few operators and heuristic for many."

    def __init__(self, default_degree, element_replace_map):
        MultiFunction.__init__(self)
        self.default_degree = default_degree
        self.element_replace_map = element_replace_map

    def constant_value(self, v):
        "Constant values are constant."
        return 0

    def constant(self, v):
        return 0

    def geometric_quantity(self, v):
        "Some geometric quantities are cellwise constant. Others are nonpolynomial and thus hard to estimate."
        if is_cellwise_constant(v):
            return 0
        else:
            # As a heuristic, just returning domain degree to bump up degree somewhat
            return v.ufl_domain().ufl_coordinate_element().degree()

    def spatial_coordinate(self, v):
        "A coordinate provides additional degrees depending on coordinate field of domain."
        return v.ufl_domain().ufl_coordinate_element().degree()

    def cell_coordinate(self, v):
        "A coordinate provides one additional degree."
        return 1

    def argument(self, v):
        """A form argument provides a degree depending on the element,
        or the default degree if the element has no degree."""
        return v.ufl_element().degree()  # FIXME: Use component to improve accuracy for mixed elements

    def coefficient(self, v):
        """A form argument provides a degree depending on the element,
        or the default degree if the element has no degree."""
        e = v.ufl_element()
        e = self.element_replace_map.get(e, e)
        d = e.degree()  # FIXME: Use component to improve accuracy for mixed elements
        if d is None:
            d = self.default_degree
        return d

    def _reduce_degree(self, v, f):
        """Reduces the estimated degree by one; used when derivatives
        are taken. Does not reduce the degree when TensorProduct elements
        or quadrilateral elements are involved."""
        if isinstance(f, int) and not isinstance(f, IrreducibleInt):
            return max(f - 1, 0)
        else:
            # if tuple, do not reduce
            return f

    def _add_degrees(self, v, *ops):
        def add_single(ops):
            if any(isinstance(o, IrreducibleInt) for o in ops):
                return IrreducibleInt(sum(ops))
            else:
                return sum(ops)

        if any(isinstance(o, tuple) for o in ops):
            # we can add a slight hack here to handle things
            # like adding 0 to (3, 3) [by expanding
            # 0 to (0, 0) when making tempops]
            tempops = [foo if isinstance(foo, tuple) else (foo, foo) for foo in ops]
            return tuple(map(add_single, zip(*tempops)))
        else:
            return add_single(ops)

    def _max_degrees(self, v, *ops):
        def max_single(ops):
            if any(isinstance(o, IrreducibleInt) for o in ops):
                return IrreducibleInt(max(ops))
            else:
                return max(ops)

        if any(isinstance(o, tuple) for o in ops):
            tempops = [foo if isinstance(foo, tuple) else (foo, foo) for foo in ops]
            return tuple(map(max_single, zip(*tempops)))
        else:
            return max_single(ops + (0,))

    def _not_handled(self, v, *args):
        error("Missing degree handler for type %s" % v._ufl_class_.__name__)

    def expr(self, v, *ops):
        "For most operators we take the max degree of its operands."
        warning("Missing degree estimation handler for type %s" % v._ufl_class_.__name__)
        return self._add_degrees(v, *ops)

    # Utility types with no degree concept
    def multi_index(self, v):
        return None

    def label(self, v):
        return None

    # Fall-through, indexing and similar types
    def reference_value(self, rv, f):
        return f

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

    def conj(self, v, a):
        return a

    def real(self, v, a):
        return a

    def imag(self, v, a):
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
        "Cell average of a function is always cellwise constant."
        return 0

    def facet_avg(self, v, a):
        "Facet average of a function is always cellwise constant."
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
        "This is a heuristic, correct if there is no "
        if a == 0:
            return a
        else:
            return a

    def division(self, v, *ops):
        "Using the sum here is a heuristic. Consider e.g. (x+1)/(x-1)."
        return self._add_degrees(v, *ops)

    def power(self, v, a, b):
        """If b is a positive integer:
        degree(a**b) == degree(a)*b
        otherwise use the heuristic
        degree(a**b) == degree(a) + 2"""
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

    def atan_2(self, v, a, b):
        """Using the heuristic
        degree(atan2(const,const)) == 0
        degree(atan2(a,b)) == max(degree(a),degree(b))+2
        which can be wildly inaccurate but at least
        gives a somewhat high integration degree.
        """
        if a or b:
            return self._add_degrees(v, self._max_degrees(v, a, b), 2)
        else:
            return self._max_degrees(v, a, b)

    def math_function(self, v, a):
        """Using the heuristic
        degree(sin(const)) == 0
        degree(sin(a)) == degree(a)+2
        which can be wildly inaccurate but at least
        gives a somewhat high integration degree.
        """
        if a:
            return self._add_degrees(v, a, 2)
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
            return self._add_degrees(v, x, 2)
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
        return self._max_degrees(v, t, f)

    def min_value(self, v, l, r):
        """Same as conditional."""
        return self._max_degrees(v, l, r)
    max_value = min_value

    def coordinate_derivative(self, v, integrand_degree, b, direction_degree, d):
        """ We use the heuristic that a shape derivative in direction V
        introduces terms V and grad(V) into the integrand. Hence we add the
        degree of the deformation to the estimate. """
        return self._add_degrees(v, integrand_degree, direction_degree)

    def expr_list(self, v, *o):
        return self._max_degrees(v, *o)

    def expr_mapping(self, v, *o):
        return self._max_degrees(v, *o)


def estimate_total_polynomial_degree(e, default_degree=1,
                                     element_replace_map={}):
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
        if not e.integrals():
            error("Got form with no integrals!")
        degrees = map_expr_dags(de, [it.integrand() for it in e.integrals()])
    elif isinstance(e, Integral):
        degrees = map_expr_dags(de, [e.integrand()])
    else:
        degrees = map_expr_dags(de, [e])
    degree = max(degrees) if degrees else default_degree
    return degree
