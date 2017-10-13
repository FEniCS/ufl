# -*- coding: utf-8 -*-
"""Algorithm to check for 'comparison' nodes
in a form when the user is in 'complex mode'"""

from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algebra import Real
from ufl.constantvalue import IntValue, FloatValue, Zero
from ufl.argument import Argument


class CheckComparisons(MultiFunction):
    """Raises an error if comparisons are done with complex quantities.

    If quantities are real, adds the Real operator to the compared quantities.

    Terminals that are real are IntValue, FloatValue, Zero, and Argument
    (even in complex FEM, the basis functions are real)
    Operations that produce reals are Abs, Real, Imag.
    Terminals default to complex, and Sqrt, Pow (defensively) imply complex.
    Otherwise, operators preserve the type of their operands.
    """
    def __init__(self):
        MultiFunction.__init__(self)
        self.nodetype = {}

    def expr(self, o, *ops):
        """Defaults expressions to complex unless they only
        act on real quantities. Overridden for specific operators.

        Rebuilds objects if necessary.
        """

        types = {self.nodetype[op] for op in ops}

        if types:
            t = "complex" if "complex" in types else "real"
        else:
            t = "complex"

        o = self.reuse_if_untouched(o, *ops)
        self.nodetype[o] = t
        return o

    def gt(self, o, *ops):
        types = {self.nodetype[op] for op in ops}

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with gt.")
        else:
            o = o._ufl_expr_reconstruct_(*map(Real, ops))
            self.nodetype[o] = "bool"
            return o

    def lt(self, o, *ops):
        types = {self.nodetype[op] for op in ops}

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with lt.")
        else:
            o = o._ufl_expr_reconstruct_(*map(Real, ops))
            self.nodetype[o] = "bool"
            return o

    def ge(self, o, *ops):
        types = {self.nodetype[op] for op in ops}

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with ge.")
        else:
            o = o._ufl_expr_reconstruct_(*map(Real, ops))
            self.nodetype[o] = "bool"
            return o

    def le(self, o, *ops):
        types = {self.nodetype[op] for op in ops}

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with le.")
        else:
            o = o._ufl_expr_reconstruct_(*map(Real, ops))
            self.nodetype[o] = "bool"
            return o

    def max_value(self, o, *ops):
        types = {self.nodetype[op] for op in ops}

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with max.")
        else:
            o = o._ufl_expr_reconstruct_(*map(Real, ops))
            self.nodetype[o] = "bool"
            return o

    def min_value(self, o, *ops):
        types = {self.nodetype[op] for op in ops}

        if "complex" in types:
            raise ComplexComparisonError("You can't compare complex numbers with min.")
        else:
            o = o._ufl_expr_reconstruct_(*map(Real, ops))
            self.nodetype[o] = "bool"
            return o

    def real(self, o, *ops):
        o = self.reuse_if_untouched(o, *ops)
        self.nodetype[o] = 'real'
        return o

    def imag(self, o, *ops):
        o = self.reuse_if_untouched(o, *ops)
        self.nodetype[o] = 'real'
        return o

    def sqrt(self, o, *ops):
        o = self.reuse_if_untouched(o, *ops)
        self.nodetype[o] = 'complex'
        return o

    def power(self, o, *ops):
        o = self.reuse_if_untouched(o, *ops)
        if float(ops[1]) < 1.0 and float(ops[1]) > 0.0:
            self.nodetype[o] = 'complex'
        elif self.nodetype[ops[0]] == 'complex':
            self.nodetype[o] = 'complex'
        else:
            self.nodetype[o] = 'real'
        return o

    def abs(self, o, *ops):
        o = self.reuse_if_untouched(o, *ops)
        self.nodetype[o] = 'real'
        return o

    def terminal(self, term, *ops):
        # default terminals to complex, except the ones we *know* are real
        if type(term) in {IntValue, FloatValue, Zero, Argument}:
            self.nodetype[term] = 'real'
        else:
            self.nodetype[term] = 'complex'
        return term


def do_comparison_check(expr):
    """Raises an error if invalid comparison nodes exist"""
    return map_integrand_dags(CheckComparisons(), expr)


class ComplexComparisonError(Exception):
    pass
