# -*- coding: utf-8 -*-
"""Algorithm to check for 'comparison' nodes
in a form when the user is in 'complex mode'"""

from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algebra import Real

class CheckComparisons(MultiFunction):
    """Raises an error if comparisons are done with complex quantities.

    If quantities are real, adds the Real operator to the compared quantities.

    Also removes unnecessary Conj of real quantities. 

    Quantities that are real are Abs, Real, Imag.
    Terminals default to complex, and Sqrt, Pow (defensively) imply complex"""
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_and_check_type

    def gt(self, o, *ops):
        ops, types = zip(*ops)

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with gt.")
        else:
            return o._ufl_expr_reconstruct_(*map(Real, *ops))

    def lt(self, o, *ops):
        ops, types = zip(*ops)

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with lt.")
        else:
            return o._ufl_expr_reconstruct_(*map(Real, *ops))

    def ge(self, o, *ops):
        ops, types = zip(*ops)

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with ge.")
        else:
            return o._ufl_expr_reconstruct_(*map(Real, *ops))

    def le(self, o, *ops):
        ops, types = zip(*ops)

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with le.")
        else:
            return o._ufl_expr_reconstruct_(*map(Real, *ops))

    def max(self, o, *ops):
        ops, types = zip(*ops)

        if "complex" in types:
            raise ComplexComparisonError("You can't order complex numbers with max.")
        else:
            return o._ufl_expr_reconstruct_(*map(Real, *ops))

    def min(self, o, *ops):
        ops, types = zip(*ops)

        if "complex" in types:
            raise ComplexComparisonError("You can't order complex numbers with min.")
        else:
            return o._ufl_expr_reconstruct_(*map(Real, *ops))

    def conj(self, o, *ops):
        ops, types = zip(*ops)

        if "complex" in types:
            print('found a complex')
            return o
        else:
            print('found a real')
            return o._ufl_expr_reconstruct_(*map(Real, *ops))

    # def real(self, o, *ops):

    # def imag(self, o, *ops):

    # def sqrt(self, o, *ops):

    # def pow(self, o, *ops):

    # def abs(self, o, *ops):

    # def terminal(self, o, *ops):
    #     # IntValue FloatValue Abs --> real
    #     # Sqrt, Pow --> imag (because we dont know the value of what's inside)
    #     # something for max and min also??


def do_comparison_check(expr):
    """Raises an error if comparison nodes exist"""
    return map_integrand_dags(CheckComparisons, expr)

class ComplexComparisonError(Exception):
    pass