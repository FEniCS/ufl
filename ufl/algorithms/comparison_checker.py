# -*- coding: utf-8 -*-
"""Algorithm to check for 'comparison' nodes
in a form when the user is in 'complex mode'"""

from ufl_legacy.corealg.multifunction import MultiFunction
from ufl_legacy.algorithms.map_integrands import map_integrand_dags
from ufl_legacy.algebra import Real
from ufl_legacy.constantvalue import RealValue, Zero
from ufl_legacy.argument import Argument
from ufl_legacy.geometry import GeometricQuantity


class CheckComparisons(MultiFunction):
    """Raises an error if comparisons are done with complex quantities.

    If quantities are real, adds the Real operator to the compared quantities.

    Terminals that are real are RealValue, Zero, and Argument
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

    def compare(self, o, *ops):
        types = {self.nodetype[op] for op in ops}

        if "complex" in types:
            raise ComplexComparisonError("Ordering undefined for complex values.")
        else:
            o = o._ufl_expr_reconstruct_(*map(Real, ops))
            self.nodetype[o] = "bool"
            return o

    gt = compare
    lt = compare
    ge = compare
    le = compare
    sign = compare

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

    def power(self, o, base, exponent):
        o = self.reuse_if_untouched(o, base, exponent)
        try:
            # Attempt to diagnose circumstances in which the result must be real.
            exponent = float(exponent)
            if self.nodetype[base] == 'real' and int(exponent) == exponent:
                self.nodetype[o] = 'real'
                return o
        except TypeError:
            pass

        self.nodetype[o] = 'complex'
        return o

    def abs(self, o, *ops):
        o = self.reuse_if_untouched(o, *ops)
        self.nodetype[o] = 'real'
        return o

    def terminal(self, term, *ops):
        # default terminals to complex, except the ones we *know* are real
        if isinstance(term, (RealValue, Zero, Argument, GeometricQuantity)):
            self.nodetype[term] = 'real'
        else:
            self.nodetype[term] = 'complex'
        return term

    def indexed(self, o, expr, multiindex):
        o = self.reuse_if_untouched(o, expr, multiindex)
        self.nodetype[o] = self.nodetype[expr]
        return o


def do_comparison_check(form):
    """Raises an error if invalid comparison nodes exist"""
    return map_integrand_dags(CheckComparisons(), form)


class ComplexComparisonError(Exception):
    pass
