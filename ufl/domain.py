"""Types for representing a geometric domain."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations  # To avoid cyclic import when type-hinting.

import numbers
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ufl.core.expr import Expr
    from ufl.finiteelement import AbstractFiniteElement  # To avoid cyclic import when type-hinting.
    from ufl.form import Form
from ufl.core.ufl_id import attach_ufl_id
from ufl.core.ufl_type import UFLObject
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.sobolevspace import H1

# Export list for ufl.classes
__all_classes__ = ["AbstractDomain", "Mesh"]


class AbstractDomain:
    """Symbolic representation of a geometric domain.

    Domain has only a geometric dimension.
    """

    _geometric_dimension: int
    _meshes: Sequence[AbstractDomain]

    def __init__(self, geometric_dimension):
        """Initialise."""
        # Validate dimensions
        if not isinstance(geometric_dimension, numbers.Integral):
            raise ValueError(
                f"Expecting integer geometric dimension, not {geometric_dimension.__class__}"
            )

        # Store validated dimensions
        self._geometric_dimension = geometric_dimension

    @property
    def geometric_dimension(self):
        """Return the dimension of the space this domain is embedded in."""
        return self._geometric_dimension

    @property
    def meshes(self):
        """Return the component meshes."""
        assert self.meshes is not None
        return self._meshes

    def __len__(self):
        """Return number of component meshes."""
        return len(self.meshes)

    def __getitem__(self, i: int) -> AbstractDomain:
        """Return i-th component mesh."""
        if i >= len(self):
            raise ValueError(f"index ({i}) >= num. component meshes ({len(self)})")
        return self.meshes[i]

    def __iter__(self):
        """Return iterable component meshes."""
        return iter(self.meshes)

    def iterable_like(self, element: AbstractFiniteElement) -> Iterable[Mesh] | MeshSequence:
        """Return iterable object that is iterable like ``element``."""
        raise NotImplementedError("iterable_like() method not implemented")

    def can_make_function_space(self, element: AbstractFiniteElement) -> bool:
        """Check whether this mesh can make a function space with ``element``."""
        raise NotImplementedError("can_make_function_space() method not implemented")


# TODO: Would it be useful to have a domain representing R^d? E.g. for
# Expression.
# class EuclideanSpace(AbstractDomain):
#     def __init__(self, geometric_dimension):
#         AbstractDomain.__init__(self, geometric_dimension, geometric_dimension)


@attach_ufl_id
class Mesh(AbstractDomain, UFLObject):
    """Symbolic representation of a mesh."""

    _ufl_id: int
    _ufl_coordinate_elements: tuple[AbstractFiniteElement, ...]

    def __init__(
        self,
        coordinate_element: Sequence[AbstractFiniteElement] | AbstractFiniteElement,
        ufl_id: Optional[int] = None,
        cargo: Any = None,
    ):
        """Initialise."""
        self._ufl_id = self._init_ufl_id(ufl_id)

        # Store reference to object that will not be used by UFL
        self._ufl_cargo = cargo
        if cargo is not None and cargo.ufl_id() != self._ufl_id:
            raise ValueError("Expecting cargo object (e.g. dolfin.Mesh) to have the same ufl_id.")

        if isinstance(coordinate_element, AbstractFiniteElement):
            self._ufl_coordinate_elements = (coordinate_element,)
        else:
            self._ufl_coordinate_elements = tuple(coordinate_element)

        # Derive dimensions from element
        (gdim,) = self._ufl_coordinate_elements[0].reference_value_shape
        for c_el in self._ufl_coordinate_elements:
            if c_el.reference_value_shape != (gdim,):
                raise ValueError("Coordinate elements must have the same reference value shape.")
        AbstractDomain.__init__(self, gdim)

    def ufl_cargo(self):
        """Return carried object that will not be used by UFL."""
        return self._ufl_cargo

    def ufl_coordinate_element(self):
        """Get the coordinate element."""
        if len(self._ufl_coordinate_elements) != 1:
            raise ValueError(
                "Cannot determine coordinate element from multiple coordinate elements."
            )
        return self._ufl_coordinate_elements[0]

    def ufl_coordinate_elements(self):
        """Get the coordinate elements."""
        return self._ufl_coordinate_elements

    def ufl_cell(self):
        """Get the cell."""
        if len(self._ufl_coordinate_elements) != 1:
            raise ValueError("Cannot determine cell type from multiple coordinate elements.")
        return self._ufl_coordinate_elements[0].cell

    def ufl_cells(self):
        """Get the cell."""
        return tuple(c_el.cell for c_el in self._ufl_coordinate_elements)

    def is_piecewise_linear_simplex_domain(self):
        """Check if the domain is a piecewise linear simplex."""

        def simplex_check(c_el):
            return c_el.embedded_superdegree <= 1 and c_el in H1 and c_el.cell.is_simplex()

        return all(simplex_check(c_el) for c_el in self._ufl_coordinate_elements)

    def __repr__(self):
        """Representation."""
        r = f"Mesh({self._ufl_coordinate_elements!r}, {self._ufl_id!r})"
        return r

    def __str__(self):
        """Format as a string."""
        return f"<Mesh #{self._ufl_id}>"

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return (self._ufl_id, self._ufl_coordinate_elements)

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return ("Mesh", renumbering[self], self._ufl_coordinate_elements)

    # NB! Dropped __lt__ here, don't want users to write 'mesh1 <
    # mesh2'.
    def _ufl_sort_key_(self):
        """UFL sort key."""
        typespecific = (self._ufl_id, self._ufl_coordinate_elements)
        return (self.geometric_dimension, "Mesh", typespecific)

    @property
    def meshes(self):
        """Return the component meshes."""
        return (self,)

    def iterable_like(self, element: AbstractFiniteElement) -> Iterable[Mesh]:
        """Return iterable object that is iterable like ``element``."""
        return iter(self for _ in range(element.num_sub_elements))

    def can_make_function_space(self, element: AbstractFiniteElement) -> bool:
        """Check whether this mesh can make a function space with ``element``."""
        # Can use with any element.
        return True


class MeshSequence(AbstractDomain, UFLObject):
    """Symbolic representation of a mixed mesh.

    This class represents a collection of meshes that, along with
    a :class:`MixedElement`, represent a mixed function space defined on
    multiple domains. This abstraction allows for defining the
    mixed function space with the conventional :class:`FunctionSpace`
    class and integrating multi-domain problems seamlessly.

    Currently, all component meshes must have the same cell type (and
    thus the same topological dimension).

    Currently, one can only perform cell integrations when
    :class:`MeshSequence`es are used.

    .. code-block:: python3

        cell = triangle
        mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1))
        mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1))
        domain = MeshSequence([mesh0, mesh1])
        elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
        elem1 = FiniteElement("Lagrange", cell, 2, (), identity_pullback, H1)
        elem = MixedElement([elem0, elem1])
        V = FunctionSpace(domain, elem)
        v = TestFunction(V)
        v0, v1 = split(v)

    """

    def __init__(self, meshes: Sequence[Mesh]):
        """Initialise."""
        if any(isinstance(m, MeshSequence) for m in meshes):
            raise NotImplementedError("""
                Currently component meshes can not include MeshSequence instances""")
        # currently only support single cell type.
        (self._ufl_cell,) = set(m.ufl_cell() for m in meshes)
        (gdim,) = set(m.geometric_dimension for m in meshes)
        AbstractDomain.__init__(self, gdim)
        self._meshes = tuple(meshes)

    def ufl_cell(self):
        """Get the cell."""
        # TODO: Might need MixedCell class for more general mixed meshes.
        return self._ufl_cell

    def __repr__(self):
        """Representation."""
        return f"MeshSequence({self._meshes!r})"

    def __str__(self):
        """Format as a string."""
        return f"<MeshSequence #{self._meshes}>"

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return ("MeshSequence", tuple(m._ufl_hash_data_() for m in self._meshes))

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return ("MeshSequence", tuple(m._ufl_signature_data_(renumbering) for m in self._meshes))

    def _ufl_sort_key_(self):
        """UFL sort key."""
        return ("MeshSequence", tuple(m._ufl_sort_key_() for m in self._meshes))

    @property
    def meshes(self):
        """Return the component meshes."""
        return self._meshes

    def iterable_like(self, element: AbstractFiniteElement) -> MeshSequence:
        """Return iterable object that is iterable like ``element``."""
        if len(self) != element.num_sub_elements:
            raise RuntimeError(f"""len(self) ({len(self)}) !=
                element.num_sub_elements ({element.num_sub_elements})""")
        return self

    def can_make_function_space(self, element: AbstractFiniteElement) -> bool:
        """Check whether this mesh can make a function space with ``element``."""
        if len(self) != element.num_sub_elements:
            return False
        else:
            return all(d.can_make_function_space(e) for d, e in zip(self, element.sub_elements))

    def ufl_coordinate_elements(self):
        """Get the coordinate elements."""
        return [e for m in self.meshes for e in m.ufl_coordinate_elements()]


def as_domain(domain):
    """Convert any valid object to an AbstractDomain type."""
    if isinstance(domain, AbstractDomain):
        # Modern UFL files and dolfin behaviour
        (domain,) = set(domain.meshes)
        return domain
    try:
        return extract_unique_domain(domain)
    except AttributeError:
        domain = domain.ufl_domain()
        (domain,) = set(domain.meshes)
        return domain


def sort_domains(domains: Sequence[AbstractDomain]):
    """Sort domains in a canonical ordering.

    Args:
        domains: Sequence of domains.

    Returns:
        `tuple` of sorted domains.

    """
    return tuple(sorted(domains, key=lambda domain: domain._ufl_sort_key_()))


def join_domains(domains: Sequence[AbstractDomain], expand_mesh_sequence: bool = True):
    """Take a list of domains and return a set with only unique domain objects.

    Args:
        domains: Sequence of domains.
        expand_mesh_sequence: If True, MeshSequence components are expanded.

    Returns:
        `set` of domains.

    """
    # Use hashing to join domains, ignore None
    joined_domains = set(domains) - set((None,))
    if expand_mesh_sequence:
        unrolled_joined_domains = set()
        for domain in joined_domains:
            unrolled_joined_domains.update(domain.meshes)
        joined_domains = unrolled_joined_domains

    if not joined_domains:
        return ()

    # Check geometric dimension compatibility
    gdims = set()
    for domain in joined_domains:
        gdims.add(domain.geometric_dimension)
    if len(gdims) != 1:
        raise ValueError("Found domains with different geometric dimensions.")

    return joined_domains


# TODO: Move these to an analysis module?


def extract_domains(expr: Expr | Form, expand_mesh_sequence: bool = True):
    """Return all domains expression is defined on.

    Args:
        expr: Expr or Form.
        expand_mesh_sequence: If True, MeshSequence components are expanded.

    Returns:
        `tuple` of domains.

    """
    from ufl.form import Form

    if isinstance(expr, Form):
        if not expand_mesh_sequence:
            raise NotImplementedError("""
                Currently, can only extract domains from a Form with expand_mesh_sequence=True""")
        # Be consistent with the numbering used in signature.
        return tuple(expr.domain_numbering().keys())
    else:
        domainlist = []
        for t in traverse_unique_terminals(expr):
            domainlist.extend(t.ufl_domains())
        return sort_domains(join_domains(domainlist, expand_mesh_sequence=expand_mesh_sequence))


def extract_unique_domain(expr, expand_mesh_sequence: bool = True):
    """Return the single unique domain expression is defined on or throw an error.

    Args:
        expr: Expr or Form.
        expand_mesh_sequence: If True, MeshSequence components are expanded.

    Returns:
        domain.

    """
    domains = extract_domains(expr, expand_mesh_sequence=expand_mesh_sequence)
    if len(domains) == 1:
        return domains[0]
    elif domains:
        raise ValueError("Found multiple domains, cannot return just one.")
    else:
        return None


def find_geometric_dimension(expr):
    """Find the geometric dimension of an expression."""
    gdims = set()
    for t in traverse_unique_terminals(expr):
        # Can have multiple domains of the same cell type.
        domains = extract_domains(t)
        if len(domains) > 0:
            (gdim,) = set(domain.geometric_dimension for domain in domains)
            gdims.add(gdim)

    if len(gdims) != 1:
        raise ValueError("Cannot determine geometric dimension from expression.")
    (gdim,) = gdims
    return gdim
