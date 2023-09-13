# -*- coding: utf-8 -*-
"""Algorithm to check for 'comparison' nodes
in a form when the user is in 'complex mode'"""

from ufl.corealg.multifunction import reuse_if_untouched
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import (Expr, GT, LT, GE, LE, MaxValue, MinValue, Sqrt, Power,
                         Abs, Terminal, Indexed)
from ufl.algebra import (Real, Imag)
from ufl.constantvalue import RealValue, Zero
from ufl.argument import Argument
from ufl.geometry import GeometricQuantity
from functools import singledispatch


@singledispatch
def check_comparisons(o, self, *ops):
    """Single-dispatch function to error if comparisons are done with complex quantities
    :arg o: UFL expression
    :arg self: class that holds the nodetype dictionary

    """
    raise AssertionError("UFL node expected, not %s" % type(o))


@check_comparisons.register(Expr)
def check_comparisons_expr(o, self, *ops):
    """Defaults expressions to complex unless they only
        act on real quantities. Overridden for specific operators.

        Rebuilds objects if necessary.
        """

    types = {self.nodetype[op] for op in ops}

    if types:
        t = "complex" if "complex" in types else "real"
    else:
        t = "complex"

    o = reuse_if_untouched(o, *ops)
    self.nodetype[o] = t
    return o


@check_comparisons.register(GT)
@check_comparisons.register(LT)
@check_comparisons.register(GE)
@check_comparisons.register(LE)
# Sign type should be here but isn't a class
def check_comparisions_compare(o, self, *ops):
    types = {self.nodetype[op] for op in ops}

    if "complex" in types:
        raise ComplexComparisonError("Ordering undefined for complex values.")
    else:
        o = o._ufl_expr_reconstruct_(*map(Real, ops))
        self.nodetype[o] = "bool"
        return o


@check_comparisons.register(MaxValue)
@check_comparisons.register(MinValue)
def check_comparisons_max_value(o, self, *ops):
    types = {self.nodetype[op] for op in ops}

    if "complex" in types:
        raise ComplexComparisonError("You can't compare complex numbers with max/min.")
    else:
        o = o._ufl_expr_reconstruct_(*map(Real, ops))
        self.nodetype[o] = "bool"
        return o


@check_comparisons.register(Real)
@check_comparisons.register(Imag)
@check_comparisons.register(Abs)
def check_comparisions_real(o, self, *ops):
    o = reuse_if_untouched(o, *ops)
    self.nodetype[o] = 'real'
    return o


@check_comparisons.register(Sqrt)
def check_comparisions_sqrt(o, self, *ops):
    o = reuse_if_untouched(o, *ops)
    self.nodetype[o] = 'complex'
    return o


@check_comparisons.register(Power)
def check_comparisions_power(o, self, base, exponent):
    o = reuse_if_untouched(o, base, exponent)
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


@check_comparisons.register(Terminal)
def check_comparisions_terminal(term, self, *ops):
    # default terminals to complex, except the ones we *know* are real
    if isinstance(term, (RealValue, Zero, Argument, GeometricQuantity)):
        self.nodetype[term] = 'real'
    else:
        self.nodetype[term] = 'complex'
    return term


@check_comparisons.register(Indexed)
def check_comparisons_indexed(o, self, expr, multiindex):
    o = reuse_if_untouched(o, expr, multiindex)
    self.nodetype[o] = self.nodetype[expr]
    return o


class CheckComparisons(object):
    """Raises an error if comparisons are done with complex quantities.

    If quantities are real, adds the Real operator to the compared quantities.

    Terminals that are real are RealValue, Zero, and Argument
    (even in complex FEM, the basis functions are real)
    Operations that produce reals are Abs, Real, Imag.
    Terminals default to complex, and Sqrt, Pow (defensively) imply complex.
    Otherwise, operators preserve the type of their operands.
    """

    def __init__(self):
        # MultiFunction.__init__(self)
        self.nodetype = {}
        self.function = check_comparisons

    def __call__(self, node, *args):
        return self.function(node, self, *args)

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

    def sign(self, o, *ops):
        # Is it even possible to use this - sign is not a ufl class.
        raise Warning("This is not implemented in the single dispatch version")
        return self.compare(o, *ops)

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
