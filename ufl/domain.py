"""Types for representing a geometric domain."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numbers

from ufl.cell import AbstractCell
from ufl.core.ufl_id import attach_ufl_id
from ufl.core.ufl_type import UFLObject
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.sobolevspace import H1

# Export list for ufl.classes
__all_classes__ = ["AbstractDomain", "Mesh", "MeshView"]


class AbstractDomain(object):
    """Symbolic representation of a geometric domain.

    Domain has only a geometric and a topological dimension.
    """

    def __init__(self, topological_dimension, geometric_dimension):
        """Initialise."""
        # Validate dimensions
        if not isinstance(geometric_dimension, numbers.Integral):
            raise ValueError(
                f"Expecting integer geometric dimension, not {geometric_dimension.__class__}"
            )
        if not isinstance(topological_dimension, numbers.Integral):
            raise ValueError(
                f"Expecting integer topological dimension, not {topological_dimension.__class__}"
            )
        if topological_dimension > geometric_dimension:
            raise ValueError("Topological dimension cannot be larger than geometric dimension.")

        # Store validated dimensions
        self._topological_dimension = topological_dimension
        self._geometric_dimension = geometric_dimension

    def geometric_dimension(self):
        """Return the dimension of the space this domain is embedded in."""
        return self._geometric_dimension

    def topological_dimension(self):
        """Return the dimension of the topology of this domain."""
        return self._topological_dimension


# TODO: Would it be useful to have a domain representing R^d? E.g. for
# Expression.
# class EuclideanSpace(AbstractDomain):
#     def __init__(self, geometric_dimension):
#         AbstractDomain.__init__(self, geometric_dimension, geometric_dimension)


@attach_ufl_id
class Mesh(AbstractDomain, UFLObject):
    """Symbolic representation of a mesh."""

    def __init__(self, coordinate_element, ufl_id=None, cargo=None):
        """Initialise."""
        self._ufl_id = self._init_ufl_id(ufl_id)

        # Store reference to object that will not be used by UFL
        self._ufl_cargo = cargo
        if cargo is not None and cargo.ufl_id() != self._ufl_id:
            raise ValueError("Expecting cargo object (e.g. dolfin.Mesh) to have the same ufl_id.")

        # No longer accepting coordinates provided as a Coefficient
        from ufl.coefficient import Coefficient

        if isinstance(coordinate_element, (Coefficient, AbstractCell)):
            raise ValueError("Expecting a coordinate element in the ufl.Mesh construct.")

        # Store coordinate element
        self._ufl_coordinate_element = coordinate_element

        # Derive dimensions from element
        (gdim,) = coordinate_element.reference_value_shape
        tdim = coordinate_element.cell.topological_dimension()
        AbstractDomain.__init__(self, tdim, gdim)

    def ufl_cargo(self):
        """Return carried object that will not be used by UFL."""
        return self._ufl_cargo

    def ufl_coordinate_element(self):
        """Get the coordinate element."""
        return self._ufl_coordinate_element

    def ufl_cell(self):
        """Get the cell."""
        return self._ufl_coordinate_element.cell

    def is_piecewise_linear_simplex_domain(self):
        """Check if the domain is a piecewise linear simplex."""
        ce = self._ufl_coordinate_element
        return ce.embedded_superdegree <= 1 and ce in H1 and self.ufl_cell().is_simplex()

    def __repr__(self):
        """Representation."""
        r = "Mesh(%s, %s)" % (repr(self._ufl_coordinate_element), repr(self._ufl_id))
        return r

    def __str__(self):
        """Format as a string."""
        return "<Mesh #%s>" % (self._ufl_id,)

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return (self._ufl_id, self._ufl_coordinate_element)

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return ("Mesh", renumbering[self], self._ufl_coordinate_element)

    # NB! Dropped __lt__ here, don't want users to write 'mesh1 <
    # mesh2'.
    def _ufl_sort_key_(self):
        """UFL sort key."""
        typespecific = (self._ufl_id, self._ufl_coordinate_element)
        return (self.geometric_dimension(), self.topological_dimension(), "Mesh", typespecific)


@attach_ufl_id
class MeshView(AbstractDomain, UFLObject):
    """Symbolic representation of a mesh."""

    def __init__(self, mesh, topological_dimension, ufl_id=None):
        """Initialise."""
        self._ufl_id = self._init_ufl_id(ufl_id)

        # Store mesh
        self._ufl_mesh = mesh

        # Derive dimensions from element
        coordinate_element = mesh.ufl_coordinate_element()
        (gdim,) = coordinate_element.value_shape
        tdim = coordinate_element.cell.topological_dimension()
        AbstractDomain.__init__(self, tdim, gdim)

    def ufl_mesh(self):
        """Get the mesh."""
        return self._ufl_mesh

    def ufl_cell(self):
        """Get the cell."""
        return self._ufl_mesh.ufl_cell()

    def is_piecewise_linear_simplex_domain(self):
        """Check if the domain is a piecewise linear simplex."""
        return self._ufl_mesh.is_piecewise_linear_simplex_domain()

    def __repr__(self):
        """Representation."""
        tdim = self.topological_dimension()
        r = "MeshView(%s, %s, %s)" % (repr(self._ufl_mesh), repr(tdim), repr(self._ufl_id))
        return r

    def __str__(self):
        """Format as a string."""
        return "<MeshView #%s of dimension %d over mesh %s>" % (
            self._ufl_id,
            self.topological_dimension(),
            self._ufl_mesh,
        )

    def _ufl_hash_data_(self):
        """UFL hash data."""
        return (self._ufl_id,) + self._ufl_mesh._ufl_hash_data_()

    def _ufl_signature_data_(self, renumbering):
        """UFL signature data."""
        return ("MeshView", renumbering[self], self._ufl_mesh._ufl_signature_data_(renumbering))

    # NB! Dropped __lt__ here, don't want users to write 'mesh1 <
    # mesh2'.
    def _ufl_sort_key_(self):
        """UFL sort key."""
        typespecific = (self._ufl_id, self._ufl_mesh)
        return (self.geometric_dimension(), self.topological_dimension(), "MeshView", typespecific)


def as_domain(domain):
    """Convert any valid object to an AbstractDomain type."""
    if isinstance(domain, AbstractDomain):
        # Modern UFL files and dolfin behaviour
        return domain

    try:
        return extract_unique_domain(domain)
    except AttributeError:
        return domain.ufl_domain()


def sort_domains(domains):
    """Sort domains in a canonical ordering."""
    return tuple(sorted(domains, key=lambda domain: domain._ufl_sort_key_()))


def join_domains(domains):
    """Take a list of domains and return a tuple with only unique domain objects.

    Checks that domains with the same id are compatible.
    """
    # Use hashing to join domains, ignore None
    domains = set(domains) - set((None,))
    if not domains:
        return ()

    # Check geometric dimension compatibility
    gdims = set()
    for domain in domains:
        gdims.add(domain.geometric_dimension())
    if len(gdims) != 1:
        raise ValueError("Found domains with different geometric dimensions.")

    return domains


# TODO: Move these to an analysis module?


def extract_domains(expr):
    """Return all domains expression is defined on."""
    domainlist = []
    for t in traverse_unique_terminals(expr):
        domainlist.extend(t.ufl_domains())
    return sorted(
        join_domains(domainlist),
        key=lambda D: (D.topological_dimension(), D.ufl_cell(), D.ufl_id()),
    )


def extract_unique_domain(expr):
    """Return the single unique domain expression is defined on or throw an error."""
    domains = extract_domains(expr)
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
        domain = extract_unique_domain(t)
        if domain is not None:
            gdims.add(domain.geometric_dimension())

    if len(gdims) != 1:
        raise ValueError("Cannot determine geometric dimension from expression.")
    (gdim,) = gdims
    return gdim
