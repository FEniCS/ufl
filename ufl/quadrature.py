"""Quadrature."""

from ufl.core.ufl_type import UFLObject


class AbstractQuadrature(UFLObject):
    """An abstract quadrature rule."""

    def __hash__(self) -> int:
        """Hash the object."""
        return hash(self._ufl_hash_data_())
