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

from ufl.corealg.traversal import traverse_unique_terminals
from ufl.log import warning, error, deprecate
from ufl.assertions import ufl_assert
from ufl.utils.formatting import istr
from ufl.utils.dicts import EmptyDict
from ufl.core.terminal import Terminal
from ufl.protocols import id_or_none
from ufl.cell import as_cell, AbstractCell, Cell, ProductCell


class Domain(object):
    """Symbolic representation of a geometrical domain.

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
        "_geometric_dimension",
        "_topological_dimension",
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
            flat_domain = arg.domain()
            self._cell = flat_domain.ufl_cell()
            self._label = flat_domain.ufl_label
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
        self._geometric_dimension = self._cell.geometric_dimension()
        self._topological_dimension = self._cell.topological_dimension()

        # Sanity checks
        ufl_assert(isinstance(self._geometric_dimension, int),
                   "Expecting integer geometric dimension.")
        ufl_assert(isinstance(self._topological_dimension, int),
                   "Expecting integer topological dimension.")
        ufl_assert(self._topological_dimension <= self._geometric_dimension,
                   "Topological dimension cannot be greater than geometric dimension.")
        ufl_assert(self._topological_dimension >= 0,
                   "Topological dimension must be non-negative.")

        if self._coordinates is not None:
            ufl_assert(isinstance(self._coordinates, Coefficient),
                        "Expecting None or Coefficient for coordinates.")
            ufl_assert(self._coordinates.domain().ufl_coordinates is None,
                        "Coordinates must be defined on a domain without coordinates of its own.")
        ufl_assert(self._label is None or isinstance(self._label, str),
                   "Expecting None or str for label.")
        ufl_assert(self._data is None or hasattr(self._data, "ufl_id"),
                   "Expecting data object to implement ufl_id().")

        # Check that we didn't get any arguments that we havent interpreted
        ufl_assert(not kwargs, "Got unused keyword arguments %s" % ', '.join(sorted(kwargs)))

    def reconstruct(self, cell=None, coordinates=None, label=None, data=None):
        "Create a new Domain object with possibly changed label or data."
        if coordinates is None:
            if cell is None:
                cell = self.ufl_cell()
            if label is None:
                label = self.ufl_label
            if data is None:
                data = self.ufl_get_mesh()
            return Domain(cell, label=label, data=data)
        else:
            ufl_assert(all((cell is None, label is None, data is None)),
                       "No other arguments allowed with coordinates.")
            return Domain(coordinates)

    def geometric_dimension(self):
        "Return the dimension of the space this domain is embedded in."
        return self._geometric_dimension

    def topological_dimension(self):
        "Return the dimension of the topology of this domain."
        return self._topological_dimension

    def is_piecewise_linear_simplex_domain(self):
        return (self.coordinate_element().degree() == 1) and self.ufl_cell().is_simplex()


    #@property # Not a property because that would break dolfin backwards compatibility when subclassing this
    def ufl_cell(self):
        "Return the cell this domain is defined in terms of."
        return self._cell

    @property
    def ufl_coordinates(self):
        "Return the coordinate vector field this domain is defined in terms of."
        return self._coordinates

    @property
    def ufl_coordinate_element(self):
        "Return the finite element of the coordinate vector field of this domain."
        x = self.ufl_coordinates
        if x is None:
            from ufl import VectorElement
            return VectorElement("Lagrange", self, 1)
        else:
            return x.element()

    @property
    def ufl_label(self): # TODO: Replace with count/ufl_id when Mesh becomes subclass
        "Return the label identifying this domain. None means no label has been set."
        return self._label

    def ufl_get_mesh(self):
        #return self # FIXME: When later subclassing this from dolfin, just return self initially, then remove this method
        return self._data


    # Deprecations
    def cell(self):
        deprecate("Domain.cell() is deprecated, please use domain.ufl_cell() method instead.")
        return self.ufl_cell()
    def coordinates(self):
        deprecate("Domain.ufl_coordinates is deprecated, please use domain.ufl_coordinates property instead.")
        return self.ufl_coordinates
    def coordinate_element(self):
        deprecate("Domain.ufl_coordinates is deprecated, please use domain.ufl_coordinates property instead.")
        return self.ufl_coordinate_element
    def label(self):
        deprecate("Domain.ufl_label is deprecated, please use domain.ufl_label property instead.")
        return self.ufl_label
    def data(self):
        deprecate("Domain.data() is deprecated, please use domain.ufl_get_mesh() instead, until this becomes obsolete in later redesign.")
        return self._data


    def signature_data(self, renumbering):
        "Signature data of domain depend on the global domain numbering."
        count = renumbering[self]
        cdata = self.ufl_cell()
        x = self.ufl_coordinates
        xdata = (None if x is None else x.signature_data(renumbering))
        return (count, cdata, xdata)

    def hash_data(self):
        # Including only id of data here.
        # If this is a problem in pydolfin, the user will just have
        # to create explicit Domain objects to avoid problems.
        # NB! This data is used in both __hash__ and __eq__.
        return (self._label,
                self._cell,
                self._coordinates, # None or a Coefficient
                id_or_none(self._data))

    def __hash__(self):
        return hash(self.hash_data())

    def __eq__(self, other):
        return type(self) == type(other) and self.hash_data() == other.hash_data()

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.hash_data() < other.hash_data()

    def __str__(self):
        if self._coordinates is None:
            c = ""
        else:
            c = " and coordinates %r" % self._coordinates
        s = (self._cell, self._label, c)
        return "<Domain built from %s with label %s%s>" % s

    def reconstruction_signature(self):
        """Format as string for evaluation as Python object.

        For use with cross language frameworks, stored in generated code
        and evaluated later in Python to reconstruct this object.

        This differs from repr in that it does not include domain
        label and data or coordinates, which must be reconstructed
        or supplied by other means.
        """
        s = (self._cell,)
        return "Domain(%r)" % s

    def __repr__(self):
        if self._coordinates is None:
            d = None if self._data is None else "<data with id %s>" % id_or_none(self._data)
            s = (self._cell, self._label, d)
            return "Domain(%r, label=%r, data=%r)" % s
        else:
            s = (self._coordinates,)
            return "Domain(%r)" % s

# --- Utility conversion functions


def as_domain(domain):
    """Convert any valid object to a Domain (in particular, cell or cellname string),
    or return domain if it is already a Domain."""
    if isinstance(domain, Domain):
        return domain
    elif hasattr(domain, "ufl_domain"):
        return domain.ufl_domain()
    else:
        return Domain(as_cell(domain))

def check_domain_compatibility(domains):
    # Validate that the domains are the same except for possibly the data
    labels = set(domain.ufl_label for domain in domains)
    ufl_assert(len(labels) == 1 or (len(labels) == 2 and None in labels),
               "Got incompatible domain labels %s in check_domain_compatibility." % (labels,))

    all_cellnames = [dom.ufl_cell().cellname() for dom in domains]
    if len(set(all_cellnames)) != 1:
        error("Cellname mismatch between domains with same label.")

    all_coordinates = set(dom.ufl_coordinates for dom in domains) - set((None,))
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
        label2domlist[domain.ufl_label].append(domain)

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
                newcoordinates = dom.ufl_coordinates
                if newcoordinates is not None:
                    ufl_assert(newcoordinates.domain().ufl_coordinates is None,
                               "A coordinate domain cannot have coordinates.")
                    break

            # Validate that coordinates match if present
            if newcoordinates is not None:
                all_coordinates = [dom.ufl_coordinates for dom in domlist]
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

def extract_domains(expr):
    domainlist = []
    for t in traverse_unique_terminals(expr):
        domainlist.extend(t.domains())
    return sorted(join_domains(domainlist))
