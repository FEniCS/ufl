# flake8: noqa
"""Pull back and push forward maps."""
# Copyright (C) 2023 Matthew Scroggs, David Ham, Garth Wells
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from abc import ABC


class NonStandardPullBackException(BaseException):
    """Exception to raise if a map is non-standard."""
    pass


class AbstractPullBack(ABC):
    """An abstract pull back."""

    def apply(self, expr: Expression) -> Expression:
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        raise NonStandardPullBackException()

    def apply_inverse(self, expr: Expression) -> Expression:
        """Apply the push forward associated with this pull back.

        Args:
            expr: A function on a reference cell

        Returns: The function pushed forward to a physical cell
        """
        raise NonStandardPullBackException()


class IdentityPullBack(AbstractPullBack):
    """The identity pull back."""

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        return expr

    def apply_inverse(self, expr):
        """Apply the push forward associated with this pull back.

        Args:
            expr: A function on a reference cell

        Returns: The function pushed forward to a physical cell
        """
        return expr


class CovariantPiola(AbstractPullBack):
    """The covariant Piola pull back."""

    def apply(self, expr):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell

        Returns: The function pulled back to the reference cell
        """
        domain = extract_unique_domain(expr)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        transform = (1.0 / detJ) * J
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(expr.ufl_shape) + 1)
        kj = (*k, j)
        return as_tensor(transform[i, j] * expr[kj], (*k, i))
