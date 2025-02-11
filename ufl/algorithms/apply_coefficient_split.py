"""Apply coefficient split.

This module contains the apply_coefficient_split function that
decomposes mixed coefficients in the given Expr into components.

"""

from __future__ import annotations

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
from ufl.core.multiindex import indices
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.tensors import as_tensor


class CoefficientSplitter(DAGTraverser):
    """DAGTraverser to split mixed coefficients."""

    def __init__(
        self,
        coefficient_split: dict,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr] | None = None,
        result_cache: dict[Expr, Expr] | None = None,
    ) -> None:
        """Initialise.

        Args:
            coefficient_split: `dict` that maps mixed coefficients to their components.
            compress: If True, ``result_cache`` will be used.
            visited_cache: cache of intermediate results; expr -> r = self.process(expr, ...).
            result_cache: cache of result objects for memory reuse, r -> r.

        """
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        self._coefficient_split = coefficient_split

    @singledispatchmethod
    def process(
        self,
        o: Expr,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> Expr:
        """Split mixed coefficients.

        Args:
            o: UFL Expression to be processed.
            reference_value: Whether :py:class:`ufl.classes.ReferenceValue` has been applied or not.
            reference_grad: Number of :py:class:`ufl.classes.ReferenceGrad`s that have been applied.
            restricted: '+', '-', or None.

        Returns:
            This ``o`` wrapped with :py:class:`ufl.classes.ReferenceValue` (if ``reference_value``),
            :py:class:`ufl.classes.ReferenceGrad` (``reference_grad`` times),
            and :py:class:`ufl.restriction.Restricted` (if
            ``restricted`` is '+' or '-'). The underlying terminal will be
            decomposed into components according to ``self._coefficient_split``.

        """
        return super().process(o)

    @process.register(Expr)
    def _(
        self,
        o: Expr,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> Expr:
        """Handle Expr."""
        return self.reuse_if_untouched(
            o,
            reference_value=reference_value,
            reference_grad=reference_grad,
            restricted=restricted,
        )

    @process.register(ReferenceValue)
    def _(
        self,
        o: ReferenceValue,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> Expr:
        """Handle ReferenceValue."""
        if reference_value:
            raise RuntimeError(f"Can not apply ReferenceValue on a ReferenceValue: got {o}")
        (op,) = o.ufl_operands
        if not op._ufl_terminal_modifiers_:
            raise ValueError(f"Must be a terminal modifier: {op!r}.")
        return self(
            op,
            reference_value=True,
            reference_grad=reference_grad,
            restricted=restricted,
        )

    @process.register(ReferenceGrad)
    def _(
        self,
        o: ReferenceGrad,
        reference_value: bool = False,
        reference_grad: int = 0,
        restricted: str | None = None,
    ) -> Expr:
        """Handle ReferenceGrad."""
        (op,) = o.ufl_operands
        if not op._ufl_terminal_modifiers_:
            raise ValueError(f"Must be a terminal modifier: {op!r}.")
        return self(
            op,
            reference_value=reference_value,
            reference_grad=reference_grad + 1,
            restricted=restricted,
        )

    @process.register(Restricted)
    def _(
        self,
        o: Restricted,
        reference_value: bool | None = False,
        reference_grad: int | None = 0,
        restricted: str | None = None,
    ) -> Expr:
        """Handle Restricted."""
        if restricted is not None:
            raise RuntimeError(f"Can not apply Restricted on a Restricted: got {o}")
        (op,) = o.ufl_operands
        if not op._ufl_terminal_modifiers_:
            raise ValueError(f"Must be a terminal modifier: {op!r}.")
        return self(
            op,
            reference_value=reference_value,
            reference_grad=reference_grad,
            restricted=o._side,
        )

    @process.register(Terminal)
    def _(
        self,
        o: Terminal,
        reference_value: bool | None = False,
        reference_grad: int = 0,
        restricted: str | None = None,
    ) -> Expr:
        """Handle Terminal."""
        return self._handle_terminal(
            o,
            reference_value=reference_value,
            reference_grad=reference_grad,
            restricted=restricted,
        )

    @process.register(Coefficient)
    def _(
        self,
        o: Coefficient,
        reference_value: bool | None = False,
        reference_grad: int = 0,
        restricted: str | None = None,
    ) -> Expr:
        """Handle Coefficient."""
        if o not in self._coefficient_split:
            return self._handle_terminal(
                o,
                reference_value=reference_value,
                reference_grad=reference_grad,
                restricted=restricted,
            )
        if not reference_value:
            raise RuntimeError("ReferenceValue expected")
        beta = indices(reference_grad)
        components = []
        for coeff in self._coefficient_split[o]:
            c = self._handle_terminal(
                coeff,
                reference_value=reference_value,
                reference_grad=reference_grad,
                restricted=restricted,
                tdim=o.ufl_element().cell.topological_dimension,
            )
            for alpha in np.ndindex(coeff.ufl_element().reference_value_shape):
                components.append(c[alpha + beta])
        (i,) = indices(1)
        return ComponentTensor(as_tensor(components)[i], MultiIndex((i,) + beta))

    def _handle_terminal(
        self,
        o: Expr,
        reference_value: bool | None = False,
        reference_grad: int = 0,
        restricted: str | None = None,
        tdim: int | None = None,
    ) -> Expr:
        """Wrap terminal as needed."""
        c = o
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
        if reference_grad > 0 and tdim is not None:
            # For CellSequence composed of cells
            # with varying topological dimensions.
            # Example:
            # MixedElement([CG(hex), CG(quad)])
            # tdim = max([3 , 2]) = 3
            # ref_grad = [
            #     [rg00, rg01, rg02],
            #     [rg10, rg11,   0 ],  # <- pad with Zeros.
            # ]
            rg_shape_inner = c.ufl_shape
            rv_shape = c.ufl_shape[:-reference_grad]
            rg_shape_outer = rv_shape + (tdim,) * reference_grad
            components = np.full(rg_shape_outer, Zero())
            for alpha in np.ndindex(rg_shape_inner):
                components[alpha] = c[alpha]
            c = as_tensor(components)
        return c


def apply_coefficient_split(expr: Expr, coefficient_split: dict) -> Expr:
    """Split mixed coefficients.

    Args:
        expr: UFL expression.
        coefficient_split: `dict` that maps mixed coefficients to their components.

    Returns:
        ``expr`` with uderlying mixed coefficients split according to ``coefficient_split``.

    """
    if not coefficient_split:
        return expr
    return CoefficientSplitter(coefficient_split)(expr)
