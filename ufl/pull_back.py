from abc import ABC, abstractmethod


class NonStandardPullBackException(BaseException):
    pass


class AbstractPullBack(ABC):
    # TODO: are inputs the wrong way round?
    def apply(self, expr: ReferenceValue) -> Expression:
        """Apply the pull back."""
        raise NonStandardPullBackException

    def apply_inverse(self, expr: Expression) -> ReferenceValue:
        """Apply the push forward associated with this pull back."""
        raise NonStandardPullBackException


class IdentityPullBack(AbstractPullBack):
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
