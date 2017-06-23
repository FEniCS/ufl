# -*- coding: utf-8 -*-
"""Algorithm to check for 'comparison' nodes
in a form when the user is in 'complex mode'"""

from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags

class CheckInvalidComparisons(MultiFunction):
    """Raises an error if comparison nodes exist"""
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def gt(self, o, a, b):
        # raise if a or b is complex
        raise ComplexComparisonError("You can't compare complex numbers with gt.")

    def lt(self, o, a, b):
        raise ComplexComparisonError("You can't compare complex numbers with lt.")

    def ge(self, o, a, b):
        raise ComplexComparisonError("You can't compare complex numbers with ge.")

    def le(self, o, a, b):
        raise ComplexComparisonError("You can't compare complex numbers with le.")

    # something for max and min also??


def do_comparison_check(expr):
    """Raises an error if comparison nodes exist"""
    return map_integrand_dags(CheckInvalidComparisons, expr)

class ComplexComparisonError(Exception):
    pass