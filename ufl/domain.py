# -*- coding: utf-8 -*-
"Types for representing a geometric domain."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from collections import defaultdict
from six import iteritems

from ufl.core.terminal import Terminal
from ufl.core.ufl_type import attach_operators_from_hash_data
from ufl.core.ufl_id import attach_ufl_id
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.log import warning, error, deprecate
from ufl.assertions import ufl_assert
from ufl.utils.formatting import istr
from ufl.utils.dicts import EmptyDict
from ufl.protocols import id_or_none
from ufl.cell import as_cell, AbstractCell, Cell, ProductCell


class AbstractDomain(object):
    """Symbolic representation of a geometric domain with only a geometric and topological dimension."""
    __slots__ = ("_topological_dimension", "_geometric_dimension")
    def __init__(self, topological_dimension, geometric_dimension):
        # Validate dimensions
        ufl_assert(isinstance(geometric_dimension, int),
                   "Expecting integer geometric dimension, not '%r'" % (geometric_dimension,))
        ufl_assert(isinstance(topological_dimension, int),
                   "Expecting integer topological dimension, not '%r'" % (topological_dimension,))
        ufl_assert(topological_dimension <= geometric_dimension,
                   "Topological dimension cannot be larger than geometric dimension.")

        # Store validated dimensions
        self._topological_dimension = topological_dimension
        self._geometric_dimension = geometric_dimension

    def geometric_dimension(self):
        "Return the dimension of the space this domain is embedded in."
        return self._geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this domain."
        return self._topological_dimension

def affine_mesh(cell):
    "Create a Mesh over a given cell type with an affine geometric parameterization."
    cell = as_cell(cell)
    gdim = cell.geometric_dimension()
    degree = 1
    coordinate_element = VectorElement("Lagrange", cell, degree, dim=gdim)
    return Mesh(coordinate_element)

# TODO: Would it be useful to have a domain representing R^d? E.g. for Expression.
#class EuclideanSpace(AbstractDomain):
#    __slots__ = ()
#    def __init__(self, geometric_dimension):
#        AbstractDomain.__init__(self, geometric_dimension, geometric_dimension)


@attach_operators_from_hash_data
@attach_ufl_id
class Mesh(AbstractDomain):
    """Symbolic representation of a mesh."""
    __slots__ = (
        "_ufl_coordinate_element",
        "_ufl_id",
        )
    def __init__(self, coordinate_element, ufl_id=None):
        self._ufl_id = self._init_ufl_id(ufl_id)

        from ufl.coefficient import Coefficient
        if isinstance(coordinate_element, Coefficient):
            error("Expecting a coordinate element in the ufl.Mesh construct.")

        # Store coordinate element
        self._ufl_coordinate_element = coordinate_element

        # Derive dimensions from element
        gdim, = coordinate_element.value_shape()
        tdim = coordinate_element.cell().topological_dimension()
        AbstractDomain.__init__(self, gdim, tdim)

    def ufl_coordinate_element(self):
        return self._ufl_coordinate_element

    def ufl_cell(self):
        return self._ufl_coordinate_element.cell()

    def is_piecewise_linear_simplex_domain(self):
        return (self._ufl_coordinate_element.degree() == 1) and self.ufl_cell().is_simplex()

    def __repr__(self):
        return "Mesh(%r, %r)" % (self._ufl_coordinate_element, self._ufl_id)

    def __str__(self):
        return "Mesh(%r, %r)" % (self._ufl_coordinate_element, self._ufl_id)

    def _ufl_hash_data_(self):
        return (self._ufl_id, self._ufl_coordinate_element)

    def _ufl_signature_data_(self, renumbering):
        return ("Mesh", renumbering[self], self._ufl_coordinate_element)

    # NB! Dropped __lt__ here as well

    def ufl_coordinates(self):
        error("Coordinate function support has been removed!")

    def ufl_get_mesh(self):
        error("Instead of calling this, just use the mesh!")
        return self

    def ufl_label(self):
        #error("Use ufl_id instead!") # FIXME:
        return "mesh_%d" % self.ufl_id()


@attach_operators_from_hash_data
class Domain(AbstractDomain): # Legacy class we're moving away from
    """Symbolic representation of a geometric domain.

    Used in the definition of geometric terminal expressions,
    finite element spaces, and integration measures.

    Takes a single positional argument which is either the
    cell of the underlying mesh

        D = Domain(triangle)

    or the coordinate field which is a vector valued Coefficient.

        P2 = VectorElement("CG", D, 2)
        x = Coefficient(P2)
        E = Domain(x)

    With the cell variant of the constructor, an optional
    label can be passed to distinguish two domains from each
    other.

        Da = Domain(cell, label="a")
        Db = Domain(cell, label="b")

    an optional data argument can also be passed, for integration
    with problem solver environments (e.g. dolfin), this is typically
    the underlying mesh.

        Da = Domain(cell, label="a", data=mesha)
        Db = Domain(cell, label="b", data=meshb)

    """
    __slots__ = (
        "_cell",
        "_coordinates",
        "_label",
        "_data",
        )
    def __init__(self, *args, **kwargs):
        # Parse positional argument, either a Cell or a Coefficient
        ufl_assert(len(args) == 1, "Only one positional argument accepted. See Domain docstring.")
        arg, = args

        # To avoid circular dependencies...
        from ufl.coefficient import Coefficient

        if isinstance(arg, AbstractCell):
            # Allow keyword arguments for label or data
            self._coordinates = None
            self._cell = arg
            self._label = kwargs.pop("label", None)
            self._data = kwargs.pop("data", None)

        elif isinstance(arg, Coefficient):
            # Disallow additional label and data, get from underlying 'flat domain'
            self._coordinates = arg
            flat_domain = arg.ufl_domain()
            self._cell = flat_domain.ufl_cell()
            self._label = flat_domain.ufl_label()
            self._data = flat_domain.ufl_get_mesh()

            # Get geometric dimension from self._coordinates shape
            gdim, = self._coordinates.ufl_shape
            if gdim != self._cell.geometric_dimension():
                warning("Using geometric dimension from coordinates!")
                self._cell = Cell(self._cell.cellname(), gdim)
            #ufl_assert(self._coordinates.ufl_shape == (self._cell.geometric_dimension(),),
            #           "Shape of coordinates %s does not match geometric dimension %d of cell." %\
            #    (self._coordinates.ufl_shape, self._cell.geometric_dimension()))
        else:
            ufl_error("Invalid first argument to Domain.")

        # Now we should have a Cell or something went wrong
        ufl_assert(isinstance(self._cell, AbstractCell), "Failed to construct a Cell from input arguments.")
        tdim = self._cell.topological_dimension()
        gdim = self._cell.geometric_dimension()
        AbstractDomain.__init__(self, tdim, gdim)

        if self._coordinates is not None:
            ufl_assert(isinstance(self._coordinates, Coefficient),
                        "Expecting None or Coefficient for coordinates.")
            ufl_assert(self._coordinates.ufl_domain().ufl_coordinates() is None,
                        "Coordinates must be defined on a domain without coordinates of its own.")
        ufl_assert(self._label is None or isinstance(self._label, str),
                   "Expecting None or str for label.")
        ufl_assert(self._data is None or hasattr(self._data, "ufl_id"),
                   "Expecting data object to implement ufl_id().")

        # Check that we didn't get any arguments that we havent interpreted
        ufl_assert(not kwargs, "Got unused keyword arguments %s" % ', '.join(sorted(kwargs)))

    def is_piecewise_linear_simplex_domain(self):
        return (self.ufl_coordinate_element().degree() == 1) and self.ufl_cell().is_simplex()

    def ufl_cell(self):
        "Return the cell this domain is defined in terms of."
        return self._cell

    def ufl_coordinates(self):
        "Return the coordinate vector field this domain is defined in terms of."
        # TODO: deprecate("Domain.ufl_coordinates() is deprecated, please use SpatialCoordinate(domain) to represent coordinates.")
        return self._coordinates

    def ufl_coordinate_element(self):
        "Return the finite element of the coordinate vector field of this domain."
        #return self._ufl_coordinate_element # TODO: Make this THE constructor argument
        x = self.ufl_coordinates()
        if x is None:
            from ufl import VectorElement
            return VectorElement("Lagrange", self.ufl_cell(), 1)
        else:
            return x.ufl_element()

    def ufl_label(self): # TODO: Replace with count/ufl_id when Mesh becomes subclass
        "Return the label identifying this domain. None means no label has been set."
        return self._label

    def ufl_get_mesh(self):
        #return self # FIXME: When later subclassing this from dolfin, just return self initially, then remove this method
        return self._data

    # Deprecations
    def cell(self):
        deprecate("Domain.cell() is deprecated, please use domain.ufl_cell() instead.")
        return self.ufl_cell()

    def coordinates(self):
        deprecate("Domain.coordinates() is deprecated, please use domain.ufl_coordinates()() instead.")
        return self.ufl_coordinates()

    def coordinate_element(self):
        deprecate("Domain.coordinate_element() is deprecated, please use domain.ufl_coordinate_element()() instead.")
        return self.ufl_coordinate_element()

    def label(self):
        deprecate("Domain.label() is deprecated, please use domain.ufl_label()() instead.")
        return self.ufl_label()

    def data(self):
        deprecate("Domain.data() is deprecated, please use domain.ufl_get_mesh() instead, until this becomes obsolete in later redesign.")
        return self._data


    def _ufl_signature_data_(self, renumbering):
        "Signature data of domain depend on the global domain numbering."
        count = renumbering[self]
        cdata = self.ufl_cell()
        x = self.ufl_coordinates()
        xdata = (None if x is None else x._ufl_signature_data_(renumbering))
        return (count, cdata, xdata)

    def _ufl_hash_data_(self):
        # Including only id of data here.
        # If this is a problem in pydolfin, the user will just have
        # to create explicit Domain objects to avoid problems.
        # NB! This data is used in both __hash__ and __eq__.
        return (self._label,
                self._cell,
                self._coordinates, # None or a Coefficient
                id_or_none(self._data))

    def __lt__(self, other):
        "Define an arbitrarily chosen but fixed sort ordering."
        if type(self) != type(other):
            return NotImplemented
        # Sort by gdim first, tdim next, then whatever's left depending on the subclass
        s = (self.geometric_dimension(), self.topological_dimension())
        o = (other.geometric_dimension(), other.topological_dimension())
        if s != o: return s < o
        return self._ufl_hash_data_() < other._ufl_hash_data_() # TODO: Safe for sorting?

    def __str__(self):
        if self._coordinates is None:
            c = ""
        else:
            c = " and coordinates %r" % self._coordinates
        s = (self._cell, self._label, c)
        return "<Domain built from %s with label %s%s>" % s

    def __repr__(self):
        if self._coordinates is None:
            d = None if self._data is None else "<data with id %s>" % id_or_none(self._data)
            s = (self._cell, self._label, d)
            return "Domain(%r, label=%r, data=%r)" % s
        else:
            s = (self._coordinates,)
            return "Domain(%r)" % s

# --- Utility conversion functions

_default_domains = {}
def default_domain(cell):
    "Create a singular default Domain from a cell, always returning the same Domain object for the same cell."
    global _default_domains
    assert isinstance(cell, AbstractCell)
    domain = _default_domains.get(cell)
    if domain is None:
        domain = Domain(cell)
        _default_domains[cell] = domain
    return domain

def as_domain(domain):
    """Convert any valid object to a Domain (in particular, cell or cellname string),
    or return domain if it is already a Domain."""
    if isinstance(domain, AbstractDomain):
        # Modern .ufl files and dolfin behaviour
        return domain
    elif hasattr(domain, "ufl_domain"):
        # If we get a dolfin.Mesh before it's changed to inherit from ufl.Mesh
        return domain.ufl_domain()
    else:
        # Legacy .ufl files # FIXME: Make this conversion in the relevant constructors closer to the user interface
        return default_domain(as_cell(domain))
    #else:
    #    error("Invalid domain %s" % (domain,))

def check_domain_compatibility(domains):
    # Validate that the domains are the same except for possibly the data
    labels = set(domain.ufl_label() for domain in domains)
    ufl_assert(len(labels) == 1 or (len(labels) == 2 and None in labels),
               "Got incompatible domain labels %s in check_domain_compatibility." % (labels,))

    all_cellnames = [dom.ufl_cell().cellname() for dom in domains]
    if len(set(all_cellnames)) != 1:
        error("Cellname mismatch between domains with same label.")

    all_coordinates = set(dom.ufl_coordinates() for dom in domains) - set((None,))
    if len(all_coordinates) > 1:
        error("Coordinates mismatch between domains with same label.")

def join_domains(domains):
    """Take a list of Domains and return a list with only unique domain objects.

    Checks that domains with the same label are compatible,
    and allows data to be None or
    """
    # Ignore Nones in input domains
    domains = [domain for domain in domains if domain is not None]

    # Build lists of domain objects with same label
    label2domlist = defaultdict(list)
    for domain in domains:
        label2domlist[domain.ufl_label()].append(domain)

    # Extract None list from this dict, map to label but only if only one exists
    if None in label2domlist:
        none_domains = {}
        if len(label2domlist) == 1:
            pass
        elif len(label2domlist) == 2:
            none_domains = label2domlist[None]
            del label2domlist[None]
            key, = list(label2domlist.keys())
            label2domlist[key].extend(none_domains)
        else:
            error("Ambiguous mapping of domains with label None to multiple domains with different labels.")
    else:
        none_domains = {}

    # Join domain data to get a list with only one domain for each label
    newdomains = []
    for label in sorted(label2domlist.keys()):
        domlist = label2domlist[label]
        if len(domlist) == 1:
            dom, = domlist
        else:
            # Validate that the domains are the same except for possibly the data
            check_domain_compatibility(domlist)

            # Pick first non-None data object
            for dom in domlist:
                newdata = dom.ufl_get_mesh()
                if newdata is not None:
                    break
            cell = dom.ufl_cell()
            gdim = dom.geometric_dimension()
            tdim = dom.topological_dimension()

            # Validate that data ids match if present
            if newdata is not None:
                data_ids = [id_or_none(dom.ufl_get_mesh()) for dom in domlist]
                data_ids = set(i for i in data_ids if i is not None)
                if len(data_ids) > 1:
                    error("Found data objects with different ids in domains with same label.")

            # Pick first non-None coordinates object
            for dom in domlist:
                newcoordinates = dom.ufl_coordinates()
                if newcoordinates is not None:
                    ufl_assert(newcoordinates.ufl_domain().ufl_coordinates() is None,
                               "A coordinate domain cannot have coordinates.")
                    break

            # Validate that coordinates match if present
            if newcoordinates is not None:
                all_coordinates = [dom.ufl_coordinates() for dom in domlist]
                all_coordinates = set(c for c in all_coordinates if c is not None)
                if len(all_coordinates) > 1:
                    error("Found different coordinates in domains with same label.")

            # Construct a new domain object with fully completed data
            if newcoordinates is not None:
                dom = Domain(newcoordinates)
            else:
                dom = Domain(cell, label=label, data=newdata)
        newdomains.append(dom)
    return tuple(newdomains)

class ProductDomain(Domain): # TODO: AbstractDomain
    """WARNING: This is work in progress, design is in no way completed."""
    __slots__ = ("_child_domains",)
    def __init__(self, domains, data=None):
        # Get the right properties of this domain
        gdim = sum(domain.geometric_dimension() for domain in domains)
        tdim = sum(domain.topological_dimension() for domain in domains)
        cell = ProductCell(*[domain.ufl_cell() for domain in domains])
        label = "product_of_%s" % "_".join(str(domain.label()) for domain in domains)

        # Initialize parent class
        Domain.__init__(self, cell, gdim, tdim, label=label, data=data)

        # Save child domains for later
        self._child_domains = tuple(domains)

    def child_domains(self):
        return self._child_domains


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
        #error("Found no domains.")
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
