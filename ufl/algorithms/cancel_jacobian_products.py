"""Cancel contractions of the Jacobian with its inverse.

This module simplifies index contractions of the Jacobian with its
inverse, before the Jacobian inverse is expanded into individual
matrix entries.  This is done in two expression traversals.

The first traversal replaces contractions of the Jacobian with its
inverse by Kronecker deltas, represented as an indexed Identity:

    IndexSum(Jacobian[a, k] * JacobianInverse[k, b] * factors, k)
        -> Identity[a, b] * factors,
    IndexSum(JacobianInverse[a, k] * Jacobian[k, b] * factors, k)
        -> Identity[a, b] * factors.

The second traversal eliminates the Identity tensors by contraction
against the remaining factors:

    IndexSum(Identity[a, k] * factors, k) -> factors[k -> a],

and folds Identity entries at fixed indices into scalar constants.

A third traversal cancels reciprocal factors within products, such as
those introduced by the Piola maps and their inverses:

    JacobianDeterminant**2 * (1 / JacobianDeterminant)**2 -> 1.

These patterns arise from Piola-mapped elements, where the pullback
inserts Jacobian factors and spatial derivatives insert Jacobian
inverse factors that cancel out, e.g. in the divergence of a
contravariant Piola-mapped function.

This pass assumes that derivatives have been expanded and that
component tensors have been removed, so that the contractions appear
as IndexSum nodes over products of indexed terminals.
"""
# Copyright (C) 2026 Pablo Brubeck
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections import defaultdict

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.remove_component_tensors import IndexReplacer
from ufl.classes import (
    Division,
    Identity,
    Indexed,
    IndexSum,
    Jacobian,
    JacobianInverse,
    Power,
    Product,
)
from ufl.constantvalue import ScalarValue, Zero, as_ufl
from ufl.core.multiindex import FixedIndex, MultiIndex
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_unique_domain


def _flatten_product(expr, factors):
    """Flatten nested Products into a list of scalar factors."""
    if isinstance(expr, Product):
        for op in expr.ufl_operands:
            _flatten_product(op, factors)
    else:
        factors.append(expr)
    return factors


def _make_product(factors):
    """Rebuild a Product from a list of scalar factors."""
    result = factors[0]
    for f in factors[1:]:
        result = Product(result, f)
    return result


class IndexSumSimplifier(MultiFunction):
    """Base class for simplifying contractions in IndexSum nodes.

    Subclasses implement the ``match`` method, which rewrites the
    product of the factors of a sum over a given index, or returns
    None when no simplification applies.  Contractions are chased
    through nested IndexSums by interchanging the order of summation.
    """

    def __init__(self):
        """Initialise."""
        MultiFunction.__init__(self)
        self._rules = {}

    expr = MultiFunction.reuse_if_untouched

    def _substitute(self, expr, k, a):
        """Replace the index k with a in expr."""
        rule = self._rules.get((k, a))
        if rule is None:
            rule = IndexReplacer({k: a})
            self._rules[(k, a)] = rule
        return map_expr_dag(rule, expr)

    def match(self, with_k, rest, k):
        """Simplify IndexSum(product(with_k + rest), k), or return None.

        Args:
            with_k: factors that have k as a free index.
            rest: factors that do not depend on k.
            k: the summation index.
        """
        raise NotImplementedError(f"match() not implemented by {type(self).__name__}.")

    def _cancel(self, factors, k):
        """Simplify IndexSum(product(factors), k), or return None.

        Only rewrites when a contraction over k cancels out; the
        expression is otherwise left alone.
        """
        with_k = []
        rest = []
        for f in factors:
            if k.count() in f.ufl_free_indices:
                with_k.append(f)
            else:
                rest.append(f)

        result = self.match(with_k, rest, k)
        if result is not None:
            return result

        if len(with_k) == 1 and isinstance(with_k[0], IndexSum):
            # Interchange sums: sum_k sum_j f(j, k) = sum_j sum_k f(j, k)
            summand, (j,) = with_k[0].ufl_operands
            inner = self._cancel(_flatten_product(summand, []), k)
            if inner is not None:
                return _make_product(rest + [self._index_sum(inner, j)])

        if len(with_k) == 2:
            # Try to push an Indexed factor into an inner IndexSum
            for f1, f2 in ((with_k[0], with_k[1]), (with_k[1], with_k[0])):
                if isinstance(f1, Indexed) and isinstance(f2, IndexSum):
                    summand, (j,) = f2.ufl_operands
                    inner = self._cancel(_flatten_product(summand, [f1]), k)
                    if inner is not None:
                        return _make_product(rest + [self._index_sum(inner, j)])
        return None

    def _index_sum(self, summand, k):
        """Construct IndexSum(summand, k), applying cancellations."""
        result = self._cancel(_flatten_product(summand, []), k)
        if result is not None:
            return result
        return IndexSum(summand, MultiIndex((k,)))

    def index_sum(self, o, summand, multiindex):
        """Simplify IndexSum."""
        (k,) = multiindex
        result = self._cancel(_flatten_product(summand, []), k)
        if result is not None:
            return result
        if o.ufl_operands[0] is summand:
            return o
        return IndexSum(summand, multiindex)


def _delta_cancellation(f1, f2, k):
    """Match a J-Jinv contraction over the index k.

    Given two scalar factors of a sum over k, return the equivalent
    Kronecker delta if they contract to one, otherwise return None.
    """
    for fa, fb in ((f1, f2), (f2, f1)):
        if not (isinstance(fa, Indexed) and isinstance(fb, Indexed)):
            continue
        A, ia = fa.ufl_operands
        B, ib = fb.ufl_operands
        if not (isinstance(A, JacobianInverse) and isinstance(B, Jacobian)):
            continue
        domain = extract_unique_domain(A)
        if domain != extract_unique_domain(B):
            continue
        ia, ib = tuple(ia), tuple(ib)
        if ia[1] == k and ib[0] == k and ia[0] != k and ib[1] != k:
            # sum_k K[a, k] * J[k, b] = Identity[a, b]
            # This holds for pseudo-inverses on immersed manifolds, as
            # K is a left inverse of J.
            tdim = domain.topological_dimension
            return Indexed(Identity(tdim), MultiIndex((ia[0], ib[1])))
        if ib[1] == k and ia[0] == k and ib[0] != k and ia[1] != k:
            # sum_k J[a, k] * K[k, b] = Identity[a, b]
            # Only valid when the Jacobian is square and invertible.
            gdim = domain.geometric_dimension
            tdim = domain.topological_dimension
            if gdim == tdim:
                return Indexed(Identity(gdim), MultiIndex((ib[0], ia[1])))
    return None


class JacobianCanceller(IndexSumSimplifier):
    """Cancel Jacobian-inverse contractions into Kronecker deltas."""

    def match(self, with_k, rest, k):
        """Replace a J-Jinv contraction over k by a Kronecker delta."""
        if len(with_k) == 2:
            delta = _delta_cancellation(with_k[0], with_k[1], k)
            if delta is not None:
                # The contraction over k is consumed by the delta
                return _make_product(rest + [delta])
        return None


def _identity_index(f, k):
    """If f is Identity[a, k] or Identity[k, a] with a != k, return a."""
    if isinstance(f, Indexed):
        A, ii = f.ufl_operands
        if isinstance(A, Identity):
            a, b = tuple(ii)
            if a == k and b != k:
                return b
            if b == k and a != k:
                return a
    return None


class IdentityEliminator(IndexSumSimplifier):
    """Eliminate Kronecker deltas represented by indexed Identity tensors."""

    def match(self, with_k, rest, k):
        """Contract an Identity factor over k against the remaining factors."""
        for i, f in enumerate(with_k):
            a = _identity_index(f, k)
            if a is not None:
                others = with_k[:i] + with_k[i + 1 :] + rest
                if not others:
                    return None
                return self._substitute(_make_product(others), k, a)
        return None

    def indexed(self, o, A, ii):
        """Fold Identity entries at fixed indices into scalar constants."""
        if isinstance(A, Identity):
            a, b = tuple(ii)
            if isinstance(a, FixedIndex) and isinstance(b, FixedIndex):
                return as_ufl(1) if int(a) == int(b) else Zero()
        return self.expr(o, A, ii)


def _as_base_exponent(f):
    """Destructure a factor into a (base, exponent) pair.

    Returns None when the factor does not have a constant real exponent.
    """
    if isinstance(f, Power):
        base, exponent = f.ufl_operands
        if isinstance(exponent, ScalarValue) and not isinstance(exponent._value, complex):
            pair = _as_base_exponent(base)
            if pair is not None:
                base, inner = pair
                return base, inner * exponent._value
        return None
    elif isinstance(f, Division):
        numerator, denominator = f.ufl_operands
        if isinstance(numerator, ScalarValue) and numerator._value == 1:
            pair = _as_base_exponent(denominator)
            if pair is not None:
                base, inner = pair
                return base, -inner
        return None
    else:
        return f, 1


def _make_power(base, exponent):
    """Construct base ** exponent for a constant real exponent."""
    if exponent == int(exponent):
        exponent = int(exponent)
    if exponent == 1:
        return base
    elif exponent == -1:
        return Division(as_ufl(1), base)
    elif exponent < 0:
        return Division(as_ufl(1), Power(base, as_ufl(-exponent)))
    else:
        return Power(base, as_ufl(exponent))


class ReciprocalCanceller(MultiFunction):
    """Cancel reciprocal factors within products."""

    expr = MultiFunction.reuse_if_untouched

    def product(self, o, a, b):
        """Cancel bases that appear with exponents of opposite signs."""
        factors = _flatten_product(b, _flatten_product(a, []))

        # Collect the exponents of each base
        exponents = defaultdict(list)
        for f in factors:
            pair = _as_base_exponent(f)
            if pair is not None:
                base, exponent = pair
                exponents[base].append(exponent)

        # Only rebuild bases that appear with exponents of opposite signs
        mixed = {
            base
            for base, es in exponents.items()
            if any(e > 0 for e in es) and any(e < 0 for e in es)
        }
        if not mixed:
            if o.ufl_operands == (a, b):
                return o
            return Product(a, b)

        parts = []
        emitted = set()
        for f in factors:
            pair = _as_base_exponent(f)
            base = pair[0] if pair is not None else None
            if base in mixed:
                if base not in emitted:
                    emitted.add(base)
                    net = sum(exponents[base])
                    if net != 0:
                        parts.append(_make_power(base, net))
            else:
                parts.append(f)

        if not parts:
            result = as_ufl(1)
        else:
            result = parts[0]
            for f in parts[1:]:
                result = Product(result, f)

        # Do not rebuild if cancellation would drop free indices
        if result.ufl_free_indices != o.ufl_free_indices:
            if o.ufl_operands == (a, b):
                return o
            return Product(a, b)
        return result


def cancel_jacobian_products(o):
    """Cancel contractions of the Jacobian with its inverse.

    Args:
        o: An Expr or Form.
    """
    o = map_integrand_dags(JacobianCanceller(), o)
    o = map_integrand_dags(IdentityEliminator(), o)
    o = map_integrand_dags(ReciprocalCanceller(), o)
    return o
