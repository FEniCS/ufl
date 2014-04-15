"Types for representing a geometric domain."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from ufl.log import warning, error, deprecate
from ufl.assertions import ufl_assert
from ufl.common import istr, EmptyDict
from ufl.terminal import Terminal
from ufl.protocols import id_or_none
from ufl.cell import as_cell, affine_cells, Cell, ProductCell


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

        if isinstance(arg, Cell):
            # Allow keyword arguments for label or data
            self._coordinates = None
            self._cell = arg
            self._label = kwargs.pop("label", None)
            self._data = kwargs.pop("data", None)

        elif isinstance(arg, Coefficient):
            # Disallow additional label and data, get from underlying 'flat domain'
            self._coordinates = arg
            flat_domain = arg.domain()
            self._cell = flat_domain.cell()
            self._label = flat_domain.label()
            self._data = flat_domain.data()

            # FIXME: Get geometric dimension from self._coordinates instead!
            ufl_assert(self._coordinates.shape() == (self._cell.geometric_dimension(),),
                       "Shape of coordinates %s does not match geometric dimension %d of cell." %\
                (self._coordinates.shape(), self._cell.geometric_dimension()))

        # Now we should have a Cell or something went wrong
        ufl_assert(isinstance(self._cell, Cell), "Failed to construct a Cell from input arguments.")
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
            ufl_assert(self._coordinates.domain().coordinates() is None,
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
                cell = self.cell()
            if label is None:
                label = self.label()
            if data is None:
                data = self.data()
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

    def cell(self):
        "Return the cell this domain is defined in terms of."
        return self._cell

    def coordinates(self):
        "Return the coordinate vector field this domain is defined in terms of."
        return self._coordinates

    def label(self):
        "Return the label identifying this domain. None means no label has been set."
        return self._label

    def is_piecewise_linear_simplex_domain(self):
        x = self.coordinates()
        return (x is None or x.element().degree() == 1) and (self.cell().cellname() in affine_cells)

    def data(self):
        "Return attached data object."
        return self._data

    def signature_data(self, domain_numbering):
        "Signature data of domain depend on the global domain numbering."
        c = self.cell()
        key = (c, self.label())
        data = (c, domain_numbering[key])
        return data

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
            c = "and coordinates %r" % self._coordinates
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

class OverlapDomain(Domain):
    """WARNING: This is work in progress, design is in no way completed."""
    __slots__ = ("_child_domains",)
    def __init__(self, domain1, domain2, label=None, data=None):
        # Check domain compatibility
        ufl_assert(domain1.cell() == domain2.cell(),
                   "Cell mismatch in overlap domain.")
        ufl_assert(domain1.geometric_dimension() == domain2.geometric_dimension(),
                   "Dimension mismatch in overlap domain.")
        ufl_assert(domain1.topological_dimension() == domain2.topological_dimension(),
                   "Dimension mismatch in overlap domain.")

        # Initialize parent class
        Domain.__init__(self, domain1.cell(), label=label, data=data)

        # Save child domains for later
        self._child_domains = (domain1, domain2)

    def child_domains(self):
        return self._child_domains

class IntersectionDomain(Domain):
    """WARNING: This is work in progress, design is in no way completed."""
    __slots__ = ("_child_domains",)
    def __init__(self, domain1, domain2, label=None, data=None):
        # Check domain compatibility
        ufl_assert(domain1.cell() == domain2.cell(),
                   "Cell mismatch in overlap domain.")
        ufl_assert(domain1.geometric_dimension() == domain2.geometric_dimension(),
                   "Dimension mismatch in overlap domain.")
        ufl_assert(domain1.topological_dimension() == domain2.topological_dimension(),
                   "Dimension mismatch in overlap domain.")

        # Get the right properties of this domain
        gdim = domain1.geometric_dimension()
        tdim = domain1.topological_dimension()-1
        cell = Cell(domain1.cell().facet_cellname(), gdim)
        ufl_assert(cell.topological_dimension() == tdim)

        # Initialize parent class
        Domain.__init__(self, cell, gdim, tdim, label=label, data=data)

        # Save child domains for later
        self._child_domains = (domain1, domain2)

    def child_domains(self):
        return self._child_domains

class ProductDomain(Domain):
    """WARNING: This is work in progress, design is in no way completed."""
    __slots__ = ("_child_domains",)
    def __init__(self, domains, data=None):
        # Get the right properties of this domain
        gdim = sum(domain.geometric_dimension() for domain in domains)
        tdim = sum(domain.topological_dimension() for domain in domains)
        cell = ProductCell(*[domain.cell() for domain in domains])
        label = "product_of_%s" % "_".join(str(domain.label()) for domain in domains)

        # Initialize parent class
        Domain.__init__(self, cell, gdim, tdim, label=label, data=data)

        # Save child domains for later
        self._child_domains = tuple(domains)

    def child_domains(self):
        return self._child_domains

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

def join_subdomain_data(subdomain_datas): # FIXME: Remove? Think it's unused now.
    newdata = {}
    for data in subdomain_datas:
        for k,v in data.iteritems():
            nv = newdata.get(k)
            if nv is None:
                # New item, just add it
                newdata[k] = v
            elif v is not None:
                id1 = id_or_none(nv)
                id2 = id_or_none(v)
                if id1 != id2:
                    error("Found multiple data objects with key %s." % k)
    return newdata

def check_domain_compatibility(domains):
    # Validate that the domains are the same except for possibly the data
    labels = set(domain.label() for domain in domains)
    ufl_assert(len(labels) == 1 or (len(labels) == 2 and None in labels),
               "Got incompatible domain labels %s in check_domain_compatibility." % (labels,))
    cell = domains[0].cell()
    coordinates = domains[0].coordinates()
    for dom in domains[1:]:
        if dom.cell() != cell:
            error("Cell mismatch between domains with same label.")
        if dom.coordinates() != coordinates:
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
        label2domlist[domain.label()].append(domain)

    # Extract None list from this dict, map to label but only if only one exists
    if None in label2domlist:
        none_domains = {}
        if len(label2domlist) == 1:
            pass
        elif len(label2domlist) == 2:
            none_domains = label2domlist[None]
            del label2domlist[None]
            key, = label2domlist.keys()
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
                newdata = dom.data()
                if newdata is not None:
                    break
            cell = dom.cell()
            gdim = dom.geometric_dimension()
            tdim = dom.topological_dimension()

            # Validate that data ids match if present
            if newdata is not None:
                data_ids = [id_or_none(dom.data()) for dom in domlist]
                data_ids = set(i for i in data_ids if i is not None)
                if len(data_ids) > 1:
                    error("Found data objects with different ids in domains with same label.")

            # Pick first non-None coordinates object
            for dom in domlist:
                newcoordinates = dom.coordinates()
                if newcoordinates is not None:
                    ufl_assert(newcoordinates.domain().coordinates() is None,
                               "A coordinate domain cannot have coordinates.")
                    break

            # Validate that coordinates match if present
            if newcoordinates is not None:
                all_coordinates = [dom.coordinates() for dom in domlist]
                all_coordinates = set(c for c in all_coordinates if c is not None)
                if len(all_coordinates) > 1:
                    error("Found different coordinates in domains with same label.")

            # Construct a new domain object with fully completed data
            dom = Domain(cell, label=label, data=newdata)
        newdomains.append(dom)
    return tuple(newdomains)

def extract_domains(expr):
    from ufl.algorithms.traversal import traverse_unique_terminals
    domainlist = []
    for t in traverse_unique_terminals(expr):
        domainlist.extend(t.domains())
    return sorted(join_domains(domainlist))
