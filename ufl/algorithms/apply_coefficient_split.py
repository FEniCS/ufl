#  using code from TSFC.

from functools import singledispatchmethod
import numpy as np
from ufl.classes import (
    Coefficient,
    ComponentTensor,
    Expr,
    MultiIndex,
    NegativeRestricted,
    PositiveRestricted,
    ReferenceGrad,
    ReferenceValue,
    Restricted,
    Terminal,
    Zero,
)
from ufl.corealg.dag_visitor import DAGVisitor
from ufl.core.multiindex import indices
from ufl.tensors import as_tensor


class CoefficientSplitter(DAGVisitor):

    def __init__(self, coefficient_split):
        """Split mixed coefficients.

        Args:
            coefficient_split: `dict` that maps mixed coefficients to their components.
            reference_value: If `ReferenceValue` has been applied.
            reference_grad: Number of `ReferenceGrad`s that have been applied.
            restricted: '+', '-', or None.
            cache: `dict` for caching DAG nodes.

        Returns:
            This node wrapped with `ReferenceValue` (if ``reference_value``),
            `ReferenceGrad` (``reference_grad`` times), and `Restricted` (if
            ``restricted`` is '+' or '-'). The underlying terminal will be
            decomposed into components according to ``coefficient_split``.

        """
        super().__init__()
        self._coefficient_split = coefficient_split

    @singledispatchmethod
    def process(self, node, *args):
        """Handle base case."""
        raise AssertionError(f"UFL node expected: got {node}")

    @process.register(Expr)
    def _(self, node, *args):
        """Handle Expr."""
        return self.reuse_if_untouched(node, *args)

    @process.register(ReferenceValue)
    def _(self, node, reference_value: bool, reference_grad: int, restricted: str):
        """Handle ReferenceValue."""
        if reference_value:
            raise RuntimeError(f"Can not apply ReferenceValue on a ReferenceValue: got {node}")
        op, = node.ufl_operands
        if not op._ufl_terminal_modifiers_:
            raise ValueError(f"Must be a terminal modifier: {op!r}.")
        return self(op, True, reference_grad, restricted)

    @process.register(ReferenceGrad)
    def _(self, node, reference_value: bool, reference_grad: int, restricted: str):
        """Handle ReferenceGrad."""
        op, = node.ufl_operands
        if not op._ufl_terminal_modifiers_:
            raise ValueError(f"Must be a terminal modifier: {op!r}.")
        return self(op, reference_value, reference_grad + 1, restricted)

    @process.register(Restricted)
    def _(self, node, reference_value: bool, reference_grad: int, restricted: str):
        """Handle Restricted."""
        if restricted is not None:
            raise RuntimeError(f"Can not apply Restricted on a Restricted: got {node}")
        op, = node.ufl_operands
        if not op._ufl_terminal_modifiers_:
            raise ValueError(f"Must be a terminal modifier: {op!r}.")
        return self(op, reference_value, reference_grad, node._side)

    @process.register(Terminal)
    def _(self, node, reference_value: bool, reference_grad: int, restricted: str):
        """Handle Terminal."""
        return self._handle_terminal(node, reference_value, reference_grad, restricted)

    @process.register(Coefficient)
    def _(self, node, reference_value: bool, reference_grad: int, restricted: str):
        """Handle Coefficient."""
        if node not in self._coefficient_split:
            return self._handle_terminal(node, reference_value, reference_grad, restricted)
        if not reference_value:
            raise RuntimeError(f"ReferenceValue expected: got {o}")
        beta = indices(reference_grad)
        components = []
        for coeff in self._coefficient_split[node]:
            c = self._handle_terminal(coeff, reference_value, reference_grad, restricted)
            for alpha in np.ndindex(coeff.ufl_element().reference_value_shape):
                components.append(c[alpha + beta])
        # Repack derivative indices to shape
        i, = indices(1)
        return ComponentTensor(as_tensor(components)[i], MultiIndex((i,) + beta))

    def _handle_terminal(self, node, reference_value: bool, reference_grad: int, restricted: str):
        c = node
        if reference_value:
            c = ReferenceValue(c)
        for _ in range(reference_grad):
            c = ReferenceGrad(c)
        if restricted == "+":
            c = PositiveRestricted(c)
        elif restricted == "-":
            c = NegativeRestricted(c)
        elif restricted is not None:
            raise RuntimeError(f"Got unknown restriction: {restricted}")
        return c


def apply_coefficient_split(expr: Expr, coefficient_split: dict):
    """Split mixed coefficients.

    Args:
        expr: UFL expression.
        coefficient_split: `dict` that maps mixed coefficients to their components.

    Returns:
        ``expr`` with uderlying mixed coefficients split according to ``coefficient_split``.

    """
    reference_value = False
    reference_grad = 0
    restricted = None
    return CoefficientSplitter(coefficient_split)(expr, reference_value, reference_grad, restricted)