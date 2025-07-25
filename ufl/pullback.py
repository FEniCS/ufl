"""Pull back and push forward maps."""
# Copyright (C) 2023 Matthew Scroggs, David Ham, Garth Wells
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Optional

import numpy as np

from ufl.core.expr import Expr
from ufl.core.multiindex import indices
from ufl.domain import AbstractDomain, MeshSequence, extract_unique_domain
from ufl.functionspace import FunctionSpace
from ufl.tensors import as_tensor

if TYPE_CHECKING:
    from ufl.finiteelement import AbstractFiniteElement as _AbstractFiniteElement

__all_classes__ = [
    "NonStandardPullbackException",
    "AbstractPullback",
    "IdentityPullback",
    "ContravariantPiola",
    "CovariantPiola",
    "L2Piola",
    "DoubleContravariantPiola",
    "DoubleCovariantPiola",
    "CovariantContravariantPiola",
    "MixedPullback",
    "SymmetricPullback",
    "PhysicalPullback",
    "CustomPullback",
    "UndefinedPullback",
]


class NonStandardPullbackException(BaseException):
    """Exception to raise if a map is non-standard."""

    pass


class AbstractPullback(ABC):
    """An abstract pull back."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a representation of the object."""

    @abstractmethod
    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """

    @abstractproperty
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""

    def apply(self, expr: Expr, domain: Optional[AbstractDomain] = None) -> Expr:
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        raise NonStandardPullbackException()


class IdentityPullback(AbstractPullback):
    """The identity pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "IdentityPullback()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return True

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        return expr

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        return element.reference_value_shape


class ContravariantPiola(AbstractPullback):
    """The contravariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "ContravariantPiola()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return False

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import Jacobian, JacobianDeterminant

        domain = domain or extract_unique_domain(expr)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        transform = (1.0 / detJ) * J
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(expr.ufl_shape) + 1)
        kj = (*k, j)
        return as_tensor(transform[i, j] * expr[kj], (*k, i))

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        gdim = domain.geometric_dimension()
        return element.reference_value_shape[:-1] + (gdim,)


class CovariantPiola(AbstractPullback):
    """The covariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "CovariantPiola()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return False

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import JacobianInverse

        domain = domain or extract_unique_domain(expr)
        K = JacobianInverse(domain)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(expr.ufl_shape) + 1)
        kj = (*k, j)
        return as_tensor(K[j, i] * expr[kj], (*k, i))

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        gdim = domain.geometric_dimension()
        return element.reference_value_shape[:-1] + (gdim,)


class L2Piola(AbstractPullback):
    """The L2 Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "L2Piola()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return False

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import JacobianDeterminant

        domain = domain or extract_unique_domain(expr)
        detJ = JacobianDeterminant(domain)
        return expr / detJ

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        return element.reference_value_shape


class DoubleContravariantPiola(AbstractPullback):
    """The double contravariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "DoubleContravariantPiola()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return False

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import Jacobian, JacobianDeterminant

        domain = domain or extract_unique_domain(expr)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
        kmn = (*k, m, n)
        return as_tensor((1.0 / detJ) ** 2 * J[i, m] * expr[kmn] * J[j, n], (*k, i, j))

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        gdim = domain.geometric_dimension()
        return element.reference_value_shape[:-2] + (gdim, gdim)


class DoubleCovariantPiola(AbstractPullback):
    """The double covariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "DoubleCovariantPiola()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return False

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import JacobianInverse

        domain = domain or extract_unique_domain(expr)
        K = JacobianInverse(domain)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
        kmn = (*k, m, n)
        return as_tensor(K[m, i] * expr[kmn] * K[n, j], (*k, i, j))

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        gdim = domain.geometric_dimension()
        return element.reference_value_shape[:-2] + (gdim, gdim)


class CovariantContravariantPiola(AbstractPullback):
    """The covariant contravariant Piola pull back."""

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "CovariantContravariantPiola()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return False

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        from ufl.classes import Jacobian, JacobianDeterminant, JacobianInverse

        domain = domain or extract_unique_domain(expr)
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        K = JacobianInverse(domain)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
        kmn = (*k, m, n)
        return as_tensor((1.0 / detJ) * K[m, i] * expr[kmn] * J[j, n], (*k, i, j))

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        gdim = domain.geometric_dimension()
        return element.reference_value_shape[:-2] + (gdim, gdim)


class MixedPullback(AbstractPullback):
    """Pull back for a mixed element."""

    def __init__(self, element: _AbstractFiniteElement):
        """Initalise.

        Args:
            element: The mixed element
        """
        self._element = element

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return f"MixedPullback({self._element!r})"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return all(e.pullback.is_identity for e in self._element.sub_elements)

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        rflat = [expr[idx] for idx in np.ndindex(expr.ufl_shape)]
        g_components = []
        offset = 0
        # For each unique piece in reference space, apply the appropriate pullback
        domain = domain or extract_unique_domain(expr, expand_mesh_sequence=False)
        if isinstance(domain, MeshSequence):
            if len(domain) != self._element.num_sub_elements:
                raise ValueError(f"""num. component meshes ({len(domain)}) !=
                    num. sub elements ({self._element.num_sub_elements})""")
        for i, subelem in enumerate(self._element.sub_elements):
            rsub = as_tensor(
                np.asarray(rflat[offset : offset + subelem.reference_value_size]).reshape(
                    subelem.reference_value_shape
                )
            )
            subdomain = domain[i] if isinstance(domain, MeshSequence) else None
            rmapped = subelem.pullback.apply(rsub, domain=subdomain)
            # Flatten into the pulled back expression for the whole thing
            g_components.extend([rmapped[idx] for idx in np.ndindex(rmapped.ufl_shape)])
            offset += subelem.reference_value_size
        # And reshape appropriately
        space = FunctionSpace(domain, self._element)
        f = as_tensor(np.asarray(g_components).reshape(space.value_shape))
        if f.ufl_shape != space.value_shape:
            raise ValueError(
                "Expecting pulled back expression with shape "
                f"'{space.value_shape}', got '{f.ufl_shape}'"
            )
        return f

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        assert element == self._element
        domains = domain.iterable_like(element)
        dim = sum(
            FunctionSpace(d, e).value_size for d, e in zip(domains, self._element.sub_elements)
        )
        return (dim,)


class SymmetricPullback(AbstractPullback):
    """Pull back for an element with symmetry."""

    def __init__(self, element: _AbstractFiniteElement, symmetry: dict[tuple[int, ...], int]):
        """Initalise.

        Args:
            element: The element
            symmetry: A dictionary mapping from the component in
                physical space to the local component
        """
        self._element = element
        self._symmetry = symmetry

        self._sub_element_value_shape = element.sub_elements[0].reference_value_shape
        for e in element.sub_elements:
            if e.reference_value_shape != self._sub_element_value_shape:
                raise ValueError("Sub-elements must all have the same value shape.")
        self._block_shape = tuple(i + 1 for i in max(symmetry.keys()))

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return f"SymmetricPullback({self._element!r}, {self._symmetry!r})"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return all(e.pullback.is_identity for e in self._element.sub_elements)

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        domain = extract_unique_domain(expr)
        space = FunctionSpace(domain, self._element)
        rflat = [expr[idx] for idx in np.ndindex(expr.ufl_shape)]
        g_components = []
        offsets = [0]
        for subelem in self._element.sub_elements:
            offsets.append(offsets[-1] + subelem.reference_value_size)
        # For each unique piece in reference space, apply the appropriate pullback
        domain = domain or extract_unique_domain(expr, expand_mesh_sequence=False)
        if isinstance(domain, MeshSequence):
            if len(domain) != self._element.num_sub_elements:
                raise ValueError(f"""num. component meshes ({len(domain)}) !=
                    num. sub elements ({self._element.num_sub_elements})""")
        for component in np.ndindex(self._block_shape):
            i = self._symmetry[component]
            subelem = self._element.sub_elements[i]
            rsub = as_tensor(
                np.asarray(rflat[offsets[i] : offsets[i + 1]]).reshape(
                    subelem.reference_value_shape
                )
            )
            subdomain = domain[i] if isinstance(domain, MeshSequence) else None
            rmapped = subelem.pullback.apply(rsub, domain=subdomain)
            # Flatten into the pulled back expression for the whole thing
            g_components.extend([rmapped[idx] for idx in np.ndindex(rmapped.ufl_shape)])
        # And reshape appropriately
        f = as_tensor(np.asarray(g_components).reshape(space.value_shape))
        if f.ufl_shape != space.value_shape:
            raise ValueError(
                f"Expecting pulled back expression with shape "
                f"'{space.value_shape}', got '{f.ufl_shape}'"
            )
        return f

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        assert isinstance(element, type(self._element))
        subelem = element.sub_elements[0]
        pvs = subelem.pullback.physical_value_shape(subelem, domain)
        return tuple(i + 1 for i in max(self._symmetry.keys())) + pvs


class PhysicalPullback(AbstractPullback):
    """Physical pull back.

    This should probably be removed.
    """

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "PhysicalPullback()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return True

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        return expr

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        return element.reference_value_shape


class CustomPullback(AbstractPullback):
    """Custom pull back.

    This should probably be removed.
    """

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "CustomPullback()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return True

    def apply(self, expr, domain=None):
        """Apply the pull back.

        Args:
            expr: A function on a physical cell
            domain: The domain on which the function is defined

        Returns: The function pulled back to the reference cell
        """
        return expr

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        return element.reference_value_shape


class UndefinedPullback(AbstractPullback):
    """Undefined pull back.

    This should probably be removed.
    """

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return "UndefinedPullback()"

    @property
    def is_identity(self) -> bool:
        """Is this pull back the identity (or the identity applied to mutliple components)."""
        return True

    def physical_value_shape(self, element, domain) -> tuple[int, ...]:
        """Get the physical value shape when this pull back is applied to an element on a domain.

        Args:
            element: The element that the pull back is applied to
            domain: The domain

        Returns:
            The value shape when the pull back is applied to the given element
        """
        return element.reference_value_shape


identity_pullback = IdentityPullback()
covariant_piola = CovariantPiola()
contravariant_piola = ContravariantPiola()
l2_piola = L2Piola()
double_covariant_piola = DoubleCovariantPiola()
double_contravariant_piola = DoubleContravariantPiola()
covariant_contravariant_piola = CovariantContravariantPiola()
physical_pullback = PhysicalPullback()
custom_pullback = CustomPullback()
undefined_pullback = UndefinedPullback()
