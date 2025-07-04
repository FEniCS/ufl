"""UFL test utils."""

import typing

from ufl.cell import Cell
from ufl.finiteelement import AbstractFiniteElement
from ufl.pullback import (
    AbstractPullback,
    IdentityPullback,
    MixedPullback,
    SymmetricPullback,
    identity_pullback,
)
from ufl.sobolevspace import H1, SobolevSpace


class FiniteElement(AbstractFiniteElement):
    """A directly defined finite element."""

    def __init__(
        self,
        family: str,
        cell: Cell,
        degree: int,
        reference_value_shape: tuple[int, ...],
        pullback: AbstractPullback,
        sobolev_space: SobolevSpace,
        sub_elements=[],
        _repr: typing.Optional[str] = None,
        _str: typing.Optional[str] = None,
        subdegree: typing.Optional[int] = None,
    ):
        """Initialise a finite element.

        This class should only be used for testing

        Args:
            family: The family name of the element
            cell: The cell on which the element is defined
            degree: The polynomial degree of the element
            reference_value_shape: The reference value shape of the element
            pullback: The pullback to use
            sobolev_space: The Sobolev space containing this element
            sub_elements: Sub elements of this element
            _repr: A string representation of this elements
            _str: A string for printing
            subdegree: The embedded subdegree of this element
        """
        if subdegree is None:
            self._subdegree = degree
        else:
            self._subdegree = subdegree
        if _repr is None:
            if len(sub_elements) > 0:
                self._repr = (
                    f'utils.FiniteElement("{family}", {cell}, {degree}, '
                    f"{reference_value_shape}, {pullback}, {sobolev_space}, {sub_elements!r})"
                )
            else:
                self._repr = (
                    f'utils.FiniteElement("{family}", {cell}, {degree}, '
                    f"{reference_value_shape}, {pullback}, {sobolev_space})"
                )
        else:
            self._repr = _repr
        if _str is None:
            self._str = f"<{family}{degree} on a {cell}>"
        else:
            self._str = _str
        self._family = family
        self._cell = cell
        self._degree = degree
        self._reference_value_shape = reference_value_shape
        self._pullback = pullback
        self._sobolev_space = sobolev_space
        self._sub_elements = sub_elements

    def __repr__(self) -> str:
        """Format as string for evaluation as Python object."""
        return self._repr

    def __str__(self) -> str:
        """Format as string for nice printing."""
        return self._str

    def __hash__(self) -> int:
        """Return a hash."""
        return hash(f"{self!r}")

    def __eq__(self, other) -> bool:
        """Check if this element is equal to another element."""
        return type(self) is type(other) and repr(self) == repr(other)

    @property
    def sobolev_space(self) -> SobolevSpace:
        """Return the underlying Sobolev space."""
        return self._sobolev_space

    @property
    def pullback(self) -> AbstractPullback:
        """Return the pullback for this element."""
        return self._pullback

    @property
    def embedded_superdegree(self) -> typing.Optional[int]:
        """Degree of the minimum degree Lagrange space that spans this element.

        This returns the degree of the lowest degree Lagrange space such
        that the polynomial space of the Lagrange space is a superspace
        of this element's polynomial space. If this element contains
        basis functions that are not in any Lagrange space, this
        function should return None.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._degree

    @property
    def embedded_subdegree(self) -> int:
        """Degree of the maximum degree Lagrange space that is spanned by this element.

        This returns the degree of the highest degree Lagrange space
        such that the polynomial space of the Lagrange space is a
        subspace of this element's polynomial space. If this element's
        polynomial space does not include the constant function, this
        function should return -1.

        Note that on a simplex cells, the polynomial space of Lagrange
        space is a complete polynomial space, but on other cells this is
        not true. For example, on quadrilateral cells, the degree 1
        Lagrange space includes the degree 2 polynomial xy.
        """
        return self._subdegree

    @property
    def cell(self) -> Cell:
        """Return the cell of the finite element."""
        return self._cell

    @property
    def reference_value_shape(self) -> tuple[int, ...]:
        """Return the shape of the value space on the reference cell."""
        return self._reference_value_shape

    @property
    def sub_elements(self) -> list:
        """Return list of sub-elements.

        This function does not recurse: ie it does not extract the
        sub-elements of sub-elements.
        """
        return self._sub_elements


class LagrangeElement(FiniteElement):
    """A Lagrange element."""

    def __init__(self, cell: Cell, degree: int, shape: tuple[int, ...] = ()):
        """Initialise."""
        super().__init__(
            "Lagrange",
            cell,
            degree,
            shape,
            identity_pullback,
            H1,
        )


class SymmetricElement(FiniteElement):
    """A symmetric finite element."""

    def __init__(
        self,
        symmetry: dict[tuple[int, ...], int],
        sub_elements: list[AbstractFiniteElement],
    ):
        """Initialise a symmetric element.

        This class should only be used for testing

        Args:
            symmetry: Map from physical components to reference components
            sub_elements: Sub-elements of this element
        """
        self._sub_elements = sub_elements
        pullback = SymmetricPullback(self, symmetry)
        reference_value_shape = (sum(e.reference_value_size for e in sub_elements),)
        degree = max(
            e.embedded_superdegree for e in sub_elements if e.embedded_superdegree is not None
        )
        cell = sub_elements[0].cell
        for e in sub_elements:
            if e.cell != cell:
                raise ValueError("All sub-elements must be defined on the same cell")
        sobolev_space = max(e.sobolev_space for e in sub_elements)

        super().__init__(
            "Symmetric element",
            cell,
            degree,
            reference_value_shape,
            pullback,
            sobolev_space,
            sub_elements=sub_elements,
            _repr=(f"utils.SymmetricElement({symmetry!r}, {sub_elements!r})"),
            _str=f"<symmetric element on a {cell}>",
        )


class MixedElement(FiniteElement):
    """A mixed element."""

    def __init__(self, sub_elements):
        """Initialise a mixed element.

        This class should only be used for testing

        Args:
            sub_elements: Sub-elements of this element
        """
        sub_elements = [MixedElement(e) if isinstance(e, list) else e for e in sub_elements]
        cell = sub_elements[0].cell
        for e in sub_elements:
            assert e.cell == cell
        degree = max(e.embedded_superdegree for e in sub_elements)
        reference_value_shape = (sum(e.reference_value_size for e in sub_elements),)
        if all(isinstance(e.pullback, IdentityPullback) for e in sub_elements):
            pullback = IdentityPullback()
        else:
            pullback = MixedPullback(self)
        sobolev_space = max(e.sobolev_space for e in sub_elements)

        super().__init__(
            "Mixed element",
            cell,
            degree,
            reference_value_shape,
            pullback,
            sobolev_space,
            sub_elements=sub_elements,
            _repr=f"utils.MixedElement({sub_elements!r})",
            _str=f"<MixedElement with {len(sub_elements)} sub-element(s)>",
        )
