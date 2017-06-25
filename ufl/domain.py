# -*- coding: utf-8 -*-
"Types for representing a geometric domain."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2009
# Modified by Marie E. Rognes 2012

import numbers

from ufl.utils.py23 import as_native_str
from ufl.utils.py23 import as_native_strings
from ufl.core.ufl_type import attach_operators_from_hash_data
from ufl.core.ufl_id import attach_ufl_id
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.log import error
from ufl.cell import as_cell, AbstractCell, TensorProductCell
from ufl.finiteelement.tensorproductelement import TensorProductElement


# Export list for ufl.classes
__all_classes__ = as_native_strings(["AbstractDomain", "Mesh", "MeshView", "TensorProductMesh"])


class AbstractDomain(object):
    """Symbolic representation of a geometric domain with only a geometric
    and topological dimension.

    """
    def __init__(self, topological_dimension, geometric_dimension):
        # Validate dimensions
        if not isinstance(geometric_dimension, numbers.Integral):
            error("Expecting integer geometric dimension, not %s" % (geometric_dimension.__class__,))
        if not isinstance(topological_dimension, numbers.Integral):
            error("Expecting integer topological dimension, not %s" % (topological_dimension.__class__,))
        if topological_dimension > geometric_dimension:
            error("Topological dimension cannot be larger than geometric dimension.")

        # Store validated dimensions
        self._topological_dimension = topological_dimension
        self._geometric_dimension = geometric_dimension

    def geometric_dimension(self):
        "Return the dimension of the space this domain is embedded in."
        return self._geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this domain."
        return self._topological_dimension

    def __unicode__(self):
        # Only in python 2
        return str(self).decode("utf-8")


# TODO: Would it be useful to have a domain representing R^d? E.g. for
# Expression.
# class EuclideanSpace(AbstractDomain):
#     def __init__(self, geometric_dimension):
#         AbstractDomain.__init__(self, geometric_dimension, geometric_dimension)


@attach_operators_from_hash_data
@attach_ufl_id
class Mesh(AbstractDomain):
    """Symbolic representation of a mesh."""
    def __init__(self, coordinate_element, ufl_id=None, cargo=None):
        self._ufl_id = self._init_ufl_id(ufl_id)

        # Store reference to object that will not be used by UFL
        self._ufl_cargo = cargo
        if cargo is not None and cargo.ufl_id() != self._ufl_id:
            error("Expecting cargo object (e.g. dolfin.Mesh) to have the same ufl_id.")

        # No longer accepting coordinates provided as a Coefficient
        from ufl.coefficient import Coefficient
        if isinstance(coordinate_element, Coefficient):
            error("Expecting a coordinate element in the ufl.Mesh construct.")

        # Accept a cell in place of an element for brevity Mesh(triangle)
        if isinstance(coordinate_element, AbstractCell):
            from ufl.finiteelement import VectorElement
            cell = coordinate_element
            coordinate_element = VectorElement("Lagrange", cell, 1,
                                               dim=cell.geometric_dimension())

        # Store coordinate element
        self._ufl_coordinate_element = coordinate_element

        # Derive dimensions from element
        gdim, = coordinate_element.value_shape()
        tdim = coordinate_element.cell().topological_dimension()
        AbstractDomain.__init__(self, tdim, gdim)

    def ufl_cargo(self):
        "Return carried object that will not be used by UFL."
        return self._ufl_cargo

    def ufl_coordinate_element(self):
        return self._ufl_coordinate_element

    def ufl_cell(self):
        return self._ufl_coordinate_element.cell()

    def is_piecewise_linear_simplex_domain(self):
        return (self._ufl_coordinate_element.degree() == 1) and self.ufl_cell().is_simplex()

    def __repr__(self):
        r = "Mesh(%s, %s)" % (repr(self._ufl_coordinate_element), repr(self._ufl_id))
        return as_native_str(r)

    def __str__(self):
        return "<Mesh #%s>" % (self._ufl_id,)

    def _ufl_hash_data_(self):
        return (self._ufl_id, self._ufl_coordinate_element)

    def _ufl_signature_data_(self, renumbering):
        return ("Mesh", renumbering[self], self._ufl_coordinate_element)

    # NB! Dropped __lt__ here, don't want users to write 'mesh1 <
    # mesh2'.
    def _ufl_sort_key_(self):
        typespecific = (self._ufl_id, self._ufl_coordinate_element)
        return (self.geometric_dimension(), self.topological_dimension(),
                "Mesh", typespecific)


@attach_operators_from_hash_data
@attach_ufl_id
class MeshView(AbstractDomain):
    """Symbolic representation of a mesh."""
    def __init__(self, mesh, topological_dimension, ufl_id=None):
        self._ufl_id = self._init_ufl_id(ufl_id)

        # Store mesh
        self._ufl_mesh = mesh

        # Derive dimensions from element
        coordinate_element = mesh.ufl_coordinate_element()
        gdim, = coordinate_element.value_shape()
        tdim = coordinate_element.cell().topological_dimension()
        AbstractDomain.__init__(self, tdim, gdim)

    def ufl_mesh(self):
        return self._ufl_mesh

    def ufl_cell(self):
        return self._ufl_mesh.ufl_cell()

    def is_piecewise_linear_simplex_domain(self):
        return self._ufl_mesh.is_piecewise_linear_simplex_domain()

    def __repr__(self):
        tdim = self.topological_dimension()
        r = "MeshView(%s, %s, %s)" % (repr(self._ufl_mesh), repr(tdim), repr(self._ufl_id))
        return as_native_str(r)

    def __str__(self):
        return "<MeshView #%s of dimension %d over mesh %s>" % (
            self._ufl_id, self.topological_dimension(), self._ufl_mesh)

    def _ufl_hash_data_(self):
        return (self._ufl_id,) + self._ufl_mesh._ufl_hash_data_()

    def _ufl_signature_data_(self, renumbering):
        return ("MeshView", renumbering[self],
                self._ufl_mesh._ufl_signature_data_(renumbering))

    # NB! Dropped __lt__ here, don't want users to write 'mesh1 <
    # mesh2'.
    def _ufl_sort_key_(self):
        typespecific = (self._ufl_id, self._ufl_mesh)
        return (self.geometric_dimension(), self.topological_dimension(),
                "MeshView", typespecific)


@attach_operators_from_hash_data
@attach_ufl_id
class TensorProductMesh(AbstractDomain):
    """Symbolic representation of a mesh."""
    def __init__(self, meshes, ufl_id=None):
        self._ufl_id = self._init_ufl_id(ufl_id)

        # TODO: Error checking of meshes
        self._ufl_meshes = meshes

        # TODO: Is this what we want to do?
        # Build cell from mesh cells
        self._ufl_cell = TensorProductCell(*[mesh.ufl_cell() for mesh in meshes])

        # TODO: Is this what we want to do?
        # Build coordinate element from mesh coordinate elements
        self._ufl_coordinate_element = TensorProductElement([mesh.ufl_coordinate_element() for mesh in meshes])

        # Derive dimensions from meshes
        gdim = sum(mesh.geometric_dimension() for mesh in meshes)
        tdim = sum(mesh.topological_dimension() for mesh in meshes)

        AbstractDomain.__init__(self, tdim, gdim)

    def ufl_coordinate_element(self):
        return self._ufl_coordinate_element

    def ufl_cell(self):
        return self._ufl_cell

    def is_piecewise_linear_simplex_domain(self):
        return False  # TODO: Any cases this is True

    def __repr__(self):
        r = "TensorProductMesh(%s, %s)" % (repr(self._ufl_meshes), repr(self._ufl_id))
        return as_native_str(r)

    def __str__(self):
        return "<TensorProductMesh #%s with meshes %s>" % (
            self._ufl_id, self._ufl_meshes)

    def _ufl_hash_data_(self):
        return (self._ufl_id,) + tuple(mesh._ufl_hash_data_() for mesh in self._ufl_meshes)

    def _ufl_signature_data_(self, renumbering):
        return ("TensorProductMesh",) + tuple(mesh._ufl_signature_data_(renumbering) for mesh in self._ufl_meshes)

    # NB! Dropped __lt__ here, don't want users to write 'mesh1 <
    # mesh2'.
    def _ufl_sort_key_(self):
        typespecific = (self._ufl_id, tuple(mesh._ufl_sort_key_() for mesh in self._ufl_meshes))
        return (self.geometric_dimension(), self.topological_dimension(),
                "TensorProductMesh", typespecific)


# --- Utility conversion functions

def affine_mesh(cell, ufl_id=None):
    "Create a Mesh over a given cell type with an affine geometric parameterization."
    from ufl.finiteelement import VectorElement
    cell = as_cell(cell)
    gdim = cell.geometric_dimension()
    degree = 1
    coordinate_element = VectorElement("Lagrange", cell, degree, dim=gdim)
    return Mesh(coordinate_element, ufl_id=ufl_id)


_default_domains = {}


def default_domain(cell):
    """Create a singular default Mesh from a cell, always returning the
    same Mesh object for the same cell.

    """
    global _default_domains
    assert isinstance(cell, AbstractCell)
    domain = _default_domains.get(cell)
    if domain is None:
        # Create one and only one affine Mesh with a negative ufl_id
        # to avoid id collision
        ufl_id = -(len(_default_domains)+1)
        domain = affine_mesh(cell, ufl_id=ufl_id)
        _default_domains[cell] = domain
    return domain


def as_domain(domain):
    """Convert any valid object to an AbstractDomain type."""
    if isinstance(domain, AbstractDomain):
        # Modern .ufl files and dolfin behaviour
        return domain
    elif hasattr(domain, "ufl_domain"):
        # If we get a dolfin.Mesh, it can provide us a corresponding
        # ufl.Mesh.  This would be unnecessary if dolfin.Mesh could
        # subclass ufl.Mesh.
        return domain.ufl_domain()
    else:
        # Legacy .ufl files
        # TODO: Make this conversion in the relevant constructors
        # closer to the user interface?
        # TODO: Make this configurable to be an error from the dolfin side?
        cell = as_cell(domain)
        return default_domain(cell)


def sort_domains(domains):
    "Sort domains in a canonical ordering."
    return tuple(sorted(domains, key=lambda domain: domain._ufl_sort_key_()))


def join_domains(domains):
    """Take a list of domains and return a tuple with only unique domain
    objects.

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
        error("Found domains with different geometric dimensions.")
    gdim, = gdims

    # Split into legacy and modern style domains
    legacy_domains = []
    modern_domains = []
    for domain in domains:
        if isinstance(domain, Mesh) and domain.ufl_id() < 0:
            assert domain.ufl_cargo() is None
            legacy_domains.append(domain)
        else:
            modern_domains.append(domain)

    # Handle legacy domains checking
    if legacy_domains:
        if modern_domains:
            error("Found both a new-style domain and a legacy default domain.\n"
                  "These should not be used interchangeably. To find the legacy\n"
                  "domain, note that it is automatically created from a cell so\n"
                  "look for constructors taking a cell.")
        return tuple(legacy_domains)

    # Handle modern domains checking (assuming correct by construction)
    return tuple(modern_domains)


# TODO: Move these to an analysis module?

def extract_domains(expr):
    "Return all domains expression is defined on."
    domainlist = []
    for t in traverse_unique_terminals(expr):
        domainlist.extend(t.ufl_domains())
    return sorted(join_domains(domainlist))


def extract_unique_domain(expr):
    "Return the single unique domain expression is defined on or throw an error."
    domains = extract_domains(expr)
    if len(domains) == 1:
        return domains[0]
    elif domains:
        error("Found multiple domains, cannot return just one.")
    else:
        return None


def find_geometric_dimension(expr):
    "Find the geometric dimension of an expression."
    gdims = set()
    for t in traverse_unique_terminals(expr):
        if hasattr(t, "ufl_domain"):
            domain = t.ufl_domain()
            if domain is not None:
                gdims.add(domain.geometric_dimension())
        if hasattr(t, "ufl_element"):
            element = t.ufl_element()
            if element is not None:
                cell = element.cell()
                if cell is not None:
                    gdims.add(cell.geometric_dimension())
    if len(gdims) != 1:
        error("Cannot determine geometric dimension from expression.")
    gdim, = gdims
    return gdim
