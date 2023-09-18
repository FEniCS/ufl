"""Pull back and push forward maps."""

from abc import ABC, abstractmethod


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
        return expr

    def apply_inverse(self, expr):
        return expr


class CovariantPiola(AbstractPullBack):
    def apply(self, expr):
        domain = extract_unique_domain(expr)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        transform = (1.0 / detJ) * J
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(expr.ufl_shape) + 1)
        kj = (*k, j)
        return as_tensor(transform[i, j] * expr[kj], (*k, i))
