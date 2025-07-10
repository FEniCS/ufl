"""Integral domains."""

from abc import ABC, abstractmethod

from ufl.cell import AbstractCell
from ufl.finiteelement import AbstractFiniteElement


class AbstractIntegralDomain(ABC):
    """Abstract base class for integral domain."""

    def __hash__(self) -> int:
        """Hash."""
        return hash(self.__repr__())

    @abstractmethod
    def __repr__(self) -> str:
        """Representation."""

    @property
    @abstractmethod
    def integral_type(self) -> str:
        """Integral type."""


class DxIntegralDomain(AbstractIntegralDomain):
    """Integral domain for dx."""

    def __init__(self, coordinate_element: AbstractFiniteElement):
        """Initialise."""
        self.coordinate_element = coordinate_element

    def __repr__(self) -> str:
        """Representation."""
        return f"DxIntegralDomain({self.coordinate_element!r})"

    @property
    def integral_type(self) -> str:
        """Integral type."""
        return "cell"


class DsIntegralDomain(AbstractIntegralDomain):
    """Integral domain for ds."""

    def __init__(self, coordinate_element: AbstractFiniteElement, facet_type: AbstractCell):
        """Initialise."""
        self.coordinate_element = coordinate_element
        self.facet_type = facet_type

    def __repr__(self) -> str:
        """Representation."""
        return f"DsIntegralDomain({self.coordinate_element!r}, {self.facet_type!r})"

    @property
    def integral_type(self) -> str:
        """Integral type."""
        return "exterior_facet"


class DSIntegralDomain(AbstractIntegralDomain):
    """Integral domain for dS."""

    def __init__(
        self,
        coordinate_element_positive_side: AbstractFiniteElement,
        coordinate_element_negative_side: AbstractFiniteElement,
        facet_type: AbstractCell,
    ):
        """Initialise."""
        self.coordinate_element_positive_side = coordinate_element_positive_side
        self.coordinate_element_negative_side = coordinate_element_negative_side
        self.facet_type = facet_type

    def __repr__(self) -> str:
        """Representation."""
        return (
            f"DSIntegralDomain({self.coordinate_element_positive_side!r}, "
            f"{self.coordinate_element_negative_side!r}, {self.facet_type!r})"
        )

    @property
    def integral_type(self) -> str:
        """Integral type."""
        return "interior_facet"
