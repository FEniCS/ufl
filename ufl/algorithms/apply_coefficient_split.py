"""Apply coefficient split.

This module contains classes and functions to split coefficients defined on mixed function spaces.
"""

import numpy as np

from ufl import indices
from ufl.checks import is_cellwise_constant
from ufl.classes import (
    Coefficient,
    ComponentTensor,
    FixedIndex,
    Form,
    Index,
    Indexed,
    ListTensor,
    MultiIndex,
    NegativeRestricted,
    PositiveRestricted,
    ReferenceGrad,
    ReferenceValue,
    Restricted,
    Zero,
)
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction, memoized_handler
from ufl.domain import extract_unique_domain
from ufl.tensors import as_tensor


class CoefficientSplitter(MultiFunction):
    """Split mixed coefficients into the components."""

    def __init__(self, coefficient_split: dict):
        """Initialise.

        Args:
            coefficient_split: map from coefficients to the components.

        """
        MultiFunction.__init__(self)
        self._coefficient_split = coefficient_split

    expr = MultiFunction.reuse_if_untouched

    def modified_terminal(self, o):
        """Handle modified terminals."""
        restriction = None
        local_derivatives = 0
        reference_value = False
        t = o
        while not t._ufl_is_terminal_:
            assert t._ufl_is_terminal_modifier_, f"Got {t!r}"
            if isinstance(t, ReferenceValue):
                assert not reference_value, "Got twice pulled back terminal!"
                reference_value = True
                (t,) = t.ufl_operands
            elif isinstance(t, ReferenceGrad):
                local_derivatives += 1
                (t,) = t.ufl_operands
            elif isinstance(t, Restricted):
                assert restriction is None, "Got twice restricted terminal!"
                restriction = t._side
                (t,) = t.ufl_operands
            elif t._ufl_terminal_modifiers_:
                raise ValueError(
                    f"Missing handler for terminal modifier type {type(t)}, object is {t!r}."
                )
            else:
                raise ValueError(f"Unexpected type {type(t)} object {t!r}.")
        if not isinstance(t, Coefficient):
            # Only split coefficients
            return o
        if t not in self._coefficient_split:
            # Only split mixed coefficients
            return o
        # Reference value expected
        if not reference_value:
            raise RuntimeError(f"ReferenceValue expected: got {o}")
        # Derivative indices
        beta = indices(local_derivatives)
        components = []
        for subcoeff in self._coefficient_split[t]:
            c = subcoeff
            # Apply terminal modifiers onto the subcoefficient
            if reference_value:
                c = ReferenceValue(c)
            for n in range(local_derivatives):
                # Return zero if expression is trivially constant. This has to
                # happen here because ReferenceGrad has no access to the
                # topological dimension of a literal zero.
                if is_cellwise_constant(c):
                    dim = extract_unique_domain(subcoeff).topological_dimension()
                    c = Zero(c.ufl_shape + (dim,), c.ufl_free_indices, c.ufl_index_dimensions)
                else:
                    c = ReferenceGrad(c)
            if restriction == "+":
                c = PositiveRestricted(c)
            elif restriction == "-":
                c = NegativeRestricted(c)
            elif restriction is not None:
                raise RuntimeError(f"Got unknown restriction: {restriction}")
            # Collect components of the subcoefficient
            for alpha in np.ndindex(subcoeff.ufl_element().reference_value_shape):
                # New modified terminal: component[alpha + beta]
                components.append(c[alpha + beta])
        # Repack derivative indices to shape
        (c,) = indices(1)
        return ComponentTensor(as_tensor(components)[c], MultiIndex((c,) + beta))

    positive_restricted = modified_terminal
    negative_restricted = modified_terminal
    reference_grad = modified_terminal
    reference_value = modified_terminal
    terminal = modified_terminal


def apply_coefficient_split(expr, coefficient_split):
    """Split mixed coefficients, so mixed elements need not be implemented.

    :arg split: A :py:class:`dict` mapping each mixed coefficient to a
                sequence of subcoefficients.  If None, calling this
                function is a no-op.

    """
    if coefficient_split is None:
        return expr
    splitter = CoefficientSplitter(coefficient_split)
    return map_expr_dag(splitter, expr)


class FixedIndexRemover(MultiFunction):
    """Handle FixedIndex."""

    def __init__(self, fimap: dict):
        """Initialise.

        Args:
           fimap: map for index replacements.

        """
        MultiFunction.__init__(self)
        self.fimap = fimap
        self._object_cache = {}

    expr = MultiFunction.reuse_if_untouched

    @memoized_handler
    def zero(self, o):
        """Handle Zero."""
        free_indices = []
        index_dimensions = []
        for i, d in zip(o.ufl_free_indices, o.ufl_index_dimensions):
            if Index(i) in self.fimap:
                ind_j = self.fimap[Index(i)]
                if not isinstance(ind_j, FixedIndex):
                    free_indices.append(ind_j.count())
                    index_dimensions.append(d)
            else:
                free_indices.append(i)
                index_dimensions.append(d)
        return Zero(
            shape=o.ufl_shape,
            free_indices=tuple(free_indices),
            index_dimensions=tuple(index_dimensions),
        )

    @memoized_handler
    def list_tensor(self, o):
        """Handle ListTensor."""
        cc = []
        for o1 in o.ufl_operands:
            comp = map_expr_dag(self, o1)
            cc.append(comp)
        return ListTensor(*cc)

    @memoized_handler
    def multi_index(self, o):
        """Handle MultiIndex."""
        return MultiIndex(tuple(self.fimap.get(i, i) for i in o.indices()))


class IndexRemover(MultiFunction):
    """Remove Indexed."""

    def __init__(self):
        """Initialise."""
        MultiFunction.__init__(self)
        self._object_cache = {}

    expr = MultiFunction.reuse_if_untouched

    @memoized_handler
    def _zero_simplify(self, o):
        """Apply simplification for Zero()."""
        (operand,) = o.ufl_operands
        operand = map_expr_dag(self, operand)
        if isinstance(operand, Zero):
            return Zero(
                shape=o.ufl_shape,
                free_indices=o.ufl_free_indices,
                index_dimensions=o.ufl_index_dimensions,
            )
        else:
            return o._ufl_expr_reconstruct_(operand)

    @memoized_handler
    def indexed(self, o):
        """Simplify indexed ComponentTensor and ListTensor."""
        o1, i1 = o.ufl_operands
        if isinstance(o1, ComponentTensor):
            o2, i2 = o1.ufl_operands
            assert len(i2.indices()) == len(i1.indices())
            fimap = dict(zip(i2.indices(), i1.indices()))
            rule = FixedIndexRemover(fimap)
            v = map_expr_dag(rule, o2)
            return map_expr_dag(self, v)
        elif isinstance(o1, ListTensor):
            if isinstance(i1[0], FixedIndex):
                o1 = o1.ufl_operands[i1[0]._value]
                if len(i1) > 1:
                    i1 = MultiIndex(i1[1:])
                    return map_expr_dag(self, Indexed(o1, i1))
                else:
                    return map_expr_dag(self, o1)
        o1 = map_expr_dag(self, o1)
        return Indexed(o1, i1)

    # Do something nicer
    positive_restricted = _zero_simplify
    negative_restricted = _zero_simplify
    reference_grad = _zero_simplify
    reference_value = _zero_simplify


def remove_component_and_list_tensors(o):
    """Remove component and list tensors."""
    if isinstance(o, Form):
        integrals = []
        for integral in o.integrals():
            integrand = remove_component_and_list_tensors(integral.integrand())
            if not isinstance(integrand, Zero):
                integrals.append(integral.reconstruct(integrand=integrand))
        return o._ufl_expr_reconstruct_(integrals)
    else:
        rule = IndexRemover()
        return map_expr_dag(rule, o)
