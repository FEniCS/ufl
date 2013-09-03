"""Algorithms for analysing argument dependencies in expressions."""

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
#
# First added:  2008-05-07
# Last changed: 2012-04-12

from ufl.assertions import ufl_assert
from ufl.classes import Expr
from ufl.algorithms.transformer import Transformer


class NotMultiLinearException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class ArgumentDependencyExtracter(Transformer):
    def __init__(self):
        Transformer.__init__(self)
        self._empty = frozenset()

    def expr(self, o, *opdeps):
        "Default for nonterminals: nonlinear in all operands."
        for d in opdeps:
            if d:
                raise NotMultiLinearException, repr(o)
        return self._empty

    def terminal(self, o):
        "Default for terminals: no dependency on Arguments."
        return self._empty

    def variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        d = self._variable_cache.get(l)
        if d is None:
            # Visit the expression our variable represents
            d = self.visit(e)
            self._variable_cache[l] = d
        return d

    def argument(self, o):
        d = frozenset((o,))
        return frozenset((d,))

    def linear(self, o, a):
        "Nonterminals that are linear with a single argument."
        return a
    grad = linear
    div = linear
    curl = linear
    transposed = linear
    trace = linear
    skew = linear
    positive_restricted = linear
    negative_restricted = linear
    cell_avg = linear
    facet_avg = linear

    def indexed(self, o, f, i):
        return f

    def spatial_derivative(self, o, a, b):
        return a

    def variable_derivative(self, o, a, b):
        if b:
            raise NotMultiLinearException, repr(o)
        return a

    def component_tensor(self, o, f, i):
        return f

    def list_tensor(self, o, *opdeps):
        "Require same dependencies for all listtensor entries."
        d = opdeps[0]
        for d2 in opdeps[1:]:
            if not d == d2:
                raise NotMultiLinearException, repr(o)
        return d

    def conditional(self, o, cond, t, f):
        "Considering EQ, NE, LE, GE, LT, GT nonlinear in this context."
        if cond or (not t == f):
            raise NotMultiLinearException, repr(o)
        return t

    def division(self, o, a, b):
        "Arguments cannot be in the denominator."
        if b:
            raise NotMultiLinearException, repr(o)
        return a

    def index_sum(self, o, f, i):
        "Index sums inherit the dependencies of their summand."
        return f

    def sum(self, o, *opdeps):
        """Sums can contain both linear and bilinear terms (we could change
        this to require that all operands have the same dependencies)."""
        # convert frozenset to a mutable set
        deps = set(opdeps[0])
        for d in opdeps[1:]:
            # d is a frozenset of frozensets
            deps.update(d)
        return frozenset(deps)

    def product(self, o, *opdeps):
        # Product operands should not depend on the same Arguments
        c = []
        adeps, bdeps = opdeps # TODO: Generalize to any number of operands using permutations
        # for each frozenset ad in the frozenset adeps
        ufl_assert(isinstance(adeps, frozenset), "Type error")
        ufl_assert(isinstance(bdeps, frozenset), "Type error")
        ufl_assert(all(isinstance(ad, frozenset) for ad in adeps), "Type error")
        ufl_assert(all(isinstance(bd, frozenset) for bd in bdeps), "Type error")
        none = frozenset((None,))
        noneset = frozenset((none,))
        if not adeps:
            adeps = noneset
        if not bdeps:
            bdeps = noneset
        for ad in adeps:
            # for each frozenset bd in the frozenset bdeps
            for bd in bdeps:
                # build frozenset cd with the combined Argument dependencies from ad and bd
                cd = (ad | bd) - none
                # build frozenset cd with the combined Argument dependencies from ad and bd
                if not len(cd) == len(ad - none) + len(bd - none):
                    raise NotMultiLinearException, repr(o)
                # remember this dependency combination
                if cd:
                    c.append(cd)
        return frozenset(c)
    inner = product
    outer = product
    dot = product
    cross = product

def extract_argument_dependencies(e):
    "Extract a set of sets of Arguments."
    ufl_assert(isinstance(e, Expr), "Expecting an Expr.")
    return ArgumentDependencyExtracter().visit(e)
