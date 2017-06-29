# -*- coding: utf-8 -*-
"""Algorithm to check for 'comparison' nodes
in a form when the user is in 'complex mode'"""

from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algebra import Real

class CheckInvalidComparisons(MultiFunction):
    """Raises an error if comparisons are done with complex quantities.

    If quantities are real, adds the Real operator to the compared quantities.

    Quantities that are real are Abs, Real, Imag.
    Terminals default to complex, and Sqrt, Pow (defensively) imply complex"""
    def __init__(self):
        MultiFunction.__init__(self)

    expr = reuse_and_check_type

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

    def real(self, o, *ops):

    def imag(self, o, *ops):

    def sqrt(self, o, *ops):

    def pow(self, o, *ops):

    def abs(self, o, *ops):

    def terminal():
        # IntValue FloatValue Abs --> real
        # Sqrt, Pow --> imag (because we dont know the value of what's inside)
        # something for max and min also??

def reuse_and_check_type(self, o, *ops):
        """Reuse object if operands are the same objects, and
        check the type of the operands.

        Use in your own subclass by setting e.g.
        ::

            expr = reuse_and_check_type

        as a default rule.
        """
        ops, types = zip(*ops)

        if types:
            t = "complex" if "complex" in types else "real"
        else:
            # Default terminals to Complex
            t = t or "complex"

        if all(a is b for a, b in zip(o.ufl_operands, ops)):
            return (o, t)
        else:
            return (o._ufl_expr_reconstruct_(*ops), t)

def do_comparison_check(expr):
    """Raises an error if comparison nodes exist"""
    return map_integrand_dags(CheckInvalidComparisons, expr)

class ComplexComparisonError(Exception):
    pass