"""Pull back and push forward maps."""
# Copyright (C) 2023 Matthew Scroggs, David Ham, Garth Wells
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import accumulate, chain
from typing import TYPE_CHECKING

import numpy

from ufl.core.expr import Expr
from ufl.core.multiindex import indices
from ufl.domain import extract_unique_domain
from ufl.tensors import as_tensor, as_vector

if TYPE_CHECKING:
    from ufl.finiteelement import AbstractFiniteElement as _AbstractFiniteElement

__all_classes__ = ["NonStandardPullBackException", "AbstractPullBack", "IdentityPullBack",
                   "ContravariantPiola", "CovariantPiola", "L2Piola", "DoubleContravariantPiola",
                   "DoubleCovariantPiola", "MixedPullBack", "SymmetricPullBack",
                   "PhysicalPullBack", "CustomPullBack", "UndefinedPullBack"]


class NonStandardPullBackException(BaseException):
    """Exception to raise if a map is non-standard."""
    pass


class AbstractPullBack(ABC):
    """An abstract pull back."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a representation of the object."""

    def apply(self, expr: Expr) -> Expr:
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        raise NonStandardPullBackException()


class IdentityPullBack(AbstractPullBack):
    """The identity pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "IdentityPullBack()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        return expr


class ContravariantPiola(AbstractPullBack):
    """The contravariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "ContravariantPiola()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import Jacobian, JacobianDeterminant

        domain = extract_unique_domain(expr)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        transform = (1.0 / detJ) * J
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(expr.ufl_shape) + 1)
        kj = (*k, j)
        return as_tensor(transform[i, j] * expr[kj], (*k, i))


class CovariantPiola(AbstractPullBack):
    """The covariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "CovariantPiola()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import JacobianInverse

        domain = extract_unique_domain(expr)
        K = JacobianInverse(domain)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(expr.ufl_shape) + 1)
        kj = (*k, j)
        return as_tensor(K[j, i] * expr[kj], (*k, i))


class L2Piola(AbstractPullBack):
    """The L2 Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "L2Piola()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import JacobianDeterminant

        domain = extract_unique_domain(expr)
        detJ = JacobianDeterminant(domain)
        return expr / detJ


class DoubleContravariantPiola(AbstractPullBack):
    """The double contravariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "DoubleContravariantPiola()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import Jacobian, JacobianDeterminant

        domain = extract_unique_domain(expr)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
        kmn = (*k, m, n)
        return as_tensor((1.0 / detJ)**2 * J[i, m] * expr[kmn] * J[j, n], (*k, i, j))


class DoubleCovariantPiola(AbstractPullBack):
    """The double covariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "DoubleCovariantPiola()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import JacobianInverse

        domain = extract_unique_domain(expr)
        K = JacobianInverse(domain)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
        kmn = (*k, m, n)
        return as_tensor(K[m, i] * expr[kmn] * K[n, j], (*k, i, j))


class MixedPullBack(AbstractPullBack):
    """Pull back for a mixed element."""

    def __init__(self, element: _AbstractFiniteElement):
        """Initalise.

        args:
            element: The mixed element
        """
        self._element = element

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return f"MixedPullBack({self._element!r})"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        rflat = [expr[idx] for idx in numpy.ndindex(expr.ufl_shape)]
        g_components = []
        offset = 0
        # For each unique piece in reference space, apply the appropriate pullback
        for subelem in self._element.sub_elements:
            vs = subelem.reference_value_size
            rsub = as_tensor(numpy.asarray(
                rflat[offset: offset + subelem.reference_value_size]
            ).reshape(subelem.reference_value_shape))
            rmapped = subelem.pull_back.apply(rsub)
            # Flatten into the pulled back expression for the whole thing
            g_components.extend([rmapped[idx] for idx in numpy.ndindex(rmapped.ufl_shape)])
            offset += subelem.reference_value_size
        # And reshape appropriately
        f = as_tensor(numpy.asarray(g_components).reshape(self._element.value_shape))
        if f.ufl_shape != self._element.value_shape:
            raise ValueError("Expecting pulled back expression with shape "
                             f"'{self._element.value_shape}', got '{f.ufl_shape}'")
        return f


class SymmetricPullBack(AbstractPullBack):
    """Pull back for an element with symmetry."""

    def __init__(self, element: _AbstractFiniteElement, symmetry: _typing.Dict[_typing.tuple[int, ...], int]):
        """Initalise.

        args:
            element: The element
            symmetry: A dictionary mapping from the component in physical space to the local component
        """
        self._element = element
        self._symmetry = symmetry

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return f"SymmetricPullBack({self._element!r}, {self._symmetry!r})"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        rflat = [expr[idx] for idx in numpy.ndindex(expr.ufl_shape)]
        g_components = []
        offsets = [0]
        for subelem in self._element.sub_elements:
            offsets.append(offsets[-1] + subelem.reference_value_size)
        # For each unique piece in reference space, apply the appropriate pullback
        for component in numpy.ndindex(self._element.value_shape):
            i = self._symmetry[component]
            subelem = self._element.sub_elements[i]
            rsub = as_tensor(numpy.asarray(
                rflat[offsets[i]:offsets[i+1]]
            ).reshape(subelem.reference_value_shape))
            print(repr(subelem))
            rmapped = subelem.pull_back.apply(rsub)
            # Flatten into the pulled back expression for the whole thing
            g_components.extend([rmapped[idx] for idx in numpy.ndindex(rmapped.ufl_shape)])
        # And reshape appropriately
        f = as_tensor(numpy.asarray(g_components).reshape(self._element.value_shape))
        if f.ufl_shape != self._element.value_shape:
            raise ValueError(f"Expecting pulled back expression with shape '{element.value_shape}', "
                             f"got '{f.ufl_shape}'")
        return f


class PhysicalPullBack(AbstractPullBack):
    """Physical pull back.

    This should probably be removed.
    """

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "PhysicalPullBack()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        return expr


class CustomPullBack(AbstractPullBack):
    """Custom pull back.

    This should probably be removed.
    """

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "CustomPullBack()"

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        return expr


class UndefinedPullBack(AbstractPullBack):
    """Undefined pull back.

    This should probably be removed.
    """

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "UndefinedPullBack()"


identity_pull_back = IdentityPullBack()
covariant_piola = CovariantPiola()
contravariant_piola = ContravariantPiola()
l2_piola = L2Piola()
double_covariant_piola = DoubleCovariantPiola()
double_contravariant_piola = DoubleContravariantPiola()
physical_pull_back = PhysicalPullBack()
custom_pull_back = CustomPullBack()
undefined_pull_back = UndefinedPullBack()
