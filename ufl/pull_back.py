"""Pull back and push forward maps."""
# Copyright (C) 2023 Matthew Scroggs, David Ham, Garth Wells
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from abc import ABC, abstractmethod
from ufl.core.expr import Expr
from ufl.core.multiindex import indices
from ufl.domain import extract_unique_domain
from ufl.tensors import as_tensor


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
covariant_poila = CovariantPiola()
contravariant_poila = ContravariantPiola()
l2_poila = L2Piola()
double_covariant_poila = DoubleCovariantPiola()
double_contravariant_poila = DoubleContravariantPiola()
physical_pull_back = PhysicalPullBack()
custom_pull_back = CustomPullBack()
undefined_pull_back = UndefinedPullBack()
