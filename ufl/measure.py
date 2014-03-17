"""The Measure class."""

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
# Modified by Anders Logg, 2008-2009
#
# First added:  2008-03-14
# Last changed: 2014-03-17

from ufl.assertions import ufl_assert
from ufl.log import error, warning
from ufl.expr import Expr
from ufl.geometry import Domain, as_domain
from ufl.checks import is_true_ufl_scalar
from ufl.constantvalue import as_ufl
from ufl.common import EmptyDict

from ufl.protocols import id_or_none, metadata_equal, metadata_hashdata

# TODO: Design a class DomainType(name, shortname, codim, num_cells, ...)?
# TODO: Improve descriptions below:

# Enumeration of valid domain types
_domain_types = [
    # === Integration over full topological dimension:
    ("cell", "dx"),                # Over a single cell
    ("macro_cell", "dE"),          # Over a group of adjacent cells (TODO: Arbitrary cell group? Where is this used?)
    #("overlap", "dO"),            # TODO: Over a cell fragment overlapping with two or more cells
    # === Integration over topological dimension - 1:
    ("exterior_facet", "ds"),      # Over facet of a single cell
    ("interior_facet", "dS"),      # Over facet between two adjacent cells
    ("surface", "dc"),             # TODO: What is this?
    #("interface", "dI"),          # Over facet fragment overlapping with two or more cells
    # === Integration over topological dimension 0
    ("point", "dP"),               # TODO: Is this over arbitrary point cloud or vertices?
    #("vertex", "dV"),             # TODO: Use this over vertices?
    # === Integration over arbitrary topological dimension
    ("quadrature_cell", "dQ"),     # Over a custom set of quadrature points and weights
    ]
domain_type_to_measure_name = dict((l,s) for l,s in _domain_types)
measure_name_to_domain_type = dict((s,l) for l,s in _domain_types)

def register_domain_type(domain_type, measure_name):
    global domain_type_to_measure_name, measure_name_to_domain_type
    ufl_assert(measure_name == domain_type_to_measure_name.get(domain_type, measure_name),
               "Domain type already added with different measure name!")
    ufl_assert(domain_type == measure_name_to_domain_type.get(measure_name, domain_type),
               "Measure name already used for another domain type!")
    domain_type_to_measure_name[domain_type] = measure_name
    measure_name_to_domain_type[measure_name] = domain_type

def as_domain_type(domain_type):
    "Map short name to long name and require a valid one."
    domain_type = domain_type.replace(" ", "_")
    domain_type = measure_name_to_domain_type.get(domain_type, domain_type)
    ufl_assert(domain_type in domain_type_to_measure_name,
               "Invalid domain_type.")
    return domain_type

def domain_types():
    "Return a tuple of all domain type strings."
    return tuple(sorted(domain_type_to_measure_name.keys()))

def measure_names():
    "Return a tuple of all measure name strings."
    return tuple(sorted(measure_name_to_domain_type.keys()))

class Measure(object):
    __slots__ = (
        "_domain_type",
        "_domain",
        "_domain_id",
        "_metadata",
        "_domain_data",
        )
    """Representation of an integration measure.

    The Measure object holds information about integration properties to be
    transferred to a Form on multiplication with a scalar expression.
    """

    # Enumeration of valid domain types (TODO: Remove these)
    CELL            = "cell"
    EXTERIOR_FACET  = "exterior_facet"
    INTERIOR_FACET  = "interior_facet"
    POINT           = "point"
    QUADRATURE_CELL = "quadrature_cell"
    MACRO_CELL      = "macro_cell"
    SURFACE         = "surface"

    def __init__(self,
                 domain_type, # "dx" etc
                 domain=None,
                 domain_id="everywhere",
                 metadata=None,
                 domain_data=None
                 ):
        """
        domain_type:
            str, one of "cell", etc.,
            or short form "dx", etc.

        domain:
            a Domain object (includes cell, dims, label, domain data)

        domain_id:
            either string "everywhere",
            a single subdomain id int,
            or tuple of ints

        metadata
            dict, with additional compiler-specific parameters
            affecting how code is generated, including parameters
            for optimization or debugging of generated code.

        domain_data
            object representing data to interpret domain_id with.
        """
        # Map short name to long name and require a valid one
        self._domain_type = as_domain_type(domain_type)

        # Check that we either have a proper Domain or none
        self._domain = None if domain is None else as_domain(domain)
        ufl_assert(self._domain is None or isinstance(self._domain, Domain),
                   "Invalid domain.")

        # Store subdomain data
        self._domain_data = domain_data
        # FIXME: Cannot require this (yet) because we currently have no way to implement ufl_id for dolfin SubDomain
        #ufl_assert(self._domain_data is None or hasattr(self._domain_data, "ufl_id"),
        #           "Invalid domain data, missing ufl_id() implementation.")

        # Accept "everywhere", single subdomain, or multiple subdomains
        ufl_assert(domain_id in ("everywhere",)
                   or isinstance(domain_id, int)
                   or (isinstance(domain_id, tuple)
                       and all(isinstance(did, int) for did in domain_id)),
                   "Invalid domain_id.")
        self._domain_id = domain_id

        # Validate compiler options are None or dict
        ufl_assert(metadata is None or isinstance(metadata, dict),
                   "Invalid metadata.")
        self._metadata = metadata or EmptyDict

    def domain_type(self):
        """Return the domain type.

        Valid domain types are "cell", "exterior_facet", "interior_facet", etc.
        """
        return self._domain_type

    def domain(self):
        """Return the domain associated with this measure.

        This may be None or a Domain object.
        """
        return self._domain

    def domain_id(self):
        "Return the domain id of this measure (integer)."
        return self._domain_id

    def metadata(self):
        """Return the integral metadata. This data is not interpreted by UFL.
        It is passed to the form compiler which can ignore it or use it to
        compile each integral of a form in a different way."""
        return self._metadata

    def reconstruct(self,
                    domain_type=None,
                    domain_id=None,
                    domain=None,
                    metadata=None,
                    domain_data=None):
        """Construct a new Measure object with some properties replaced with new values.

        Example:
            <dm = Measure instance>
            b = dm.reconstruct(domain_id=2)
            c = dm.reconstruct(metadata={ "quadrature_degree": 3 })

        Used by the call operator, so this is equivalent:
            b = dm(2)
            c = dm(0, { "quadrature_degree": 3 })
        """
        if domain_id is None:
            domain_id = self.domain_id()
        if domain is None:
            domain = self.domain()
        if metadata is None:
            metadata = self.metadata()
        if domain_data is None:
            domain_data = self.domain_data()
        return Measure(self.domain_type(),
                       domain=domain, domain_id=domain_id,
                       metadata=metadata, domain_data=domain_data)

    def domain_data(self):
        """Return the integral domain_data. This data is not interpreted by UFL.
        Its intension is to give a context in which the domain id is interpreted."""
        return self._domain_data

    def __call__(self, domain_id=None, metadata=None, domain=None, domain_data=None):
        """Reconfigure measure with new domain specification or metadata."""
        # Note: Keeping the order of arguments here (domain_id, metadata) for backwards
        # compatibility, because some tutorials write e.g. dx(0, {...}) to set metadata

        # Let syntax dx() mean integral over everywhere
        args = (domain_id, metadata, domain, domain_data)
        if all(arg is None for arg in args):
            return self.reconstruct(domain_id="everywhere")

        # Let syntax dx(domain) or dx(domain, metadata) mean integral over entire domain.
        # To do this we need to hijack the first argument:
        if domain_id is not None and (isinstance(domain_id, Domain) or hasattr(domain_id, 'ufl_domain')):
            ufl_assert(domain is None, "Ambiguous: setting domain both as keyword argument and first argument.")
            domain_id, domain = "everywhere", as_domain(domain_id)

        # If we get any keywords, use them to reconstruct Measure.
        # Note that if only one argument is given, it is the domain_id,
        # e.g. dx(3) == dx(domain_id=3)
        return self.reconstruct(domain_id=domain_id, domain=domain, metadata=metadata, domain_data=domain_data)

    def __getitem__(self, data):
        """This operator supports legacy syntax in python dolfin programs.

        The implementation makes assumptions on the type of data,
        namely that it is a dolfin MeshFunction with a member mesh()
        which returns a dolfin Mesh.

        The intention is to deprecase and remove this operator at
        some later point. Please attach your domain data to a Domain
        object instead of using the ds[data] syntax.

        The old documentation reads:
        Return a new Measure for same integration type with an attached
        context for interpreting domain ids. By default this new Measure
        integrates over everywhere, but it can be restricted with a domain id
        as usual. Example: dx = dx[boundaries]; L = f*v*dx + g*v+dx(1).
        """
        return self.reconstruct(domain_data=data)

    def __str__(self):
        global domain_type_to_measure_name
        d = domain_type_to_measure_name[self._domain_type]
        args = []

        if self._domain_id is not None:
            args.append("domain_id=%s" % (self._domain_id,))
        if self._domain is not None:
            args.append("domain=%s" % (self._domain,))
        if self._metadata: # Stored as EmptyDict if None
            args.append("metadata=%s" % (self._metadata,))
        if self._domain_data is not None:
            args.append("domain_data=%s" % (self._domain_data,))

        return "%s(%s)" % (dm, ', '.join(args))

    def __repr__(self):
        "Return a repr string for this Measure."
        global domain_type_to_measure_name
        d = domain_type_to_measure_name[self._domain_type]

        args = []
        args.append(repr(self._domain_type))

        if self._domain_id is not None:
            args.append("domain_id=%r" % (self._domain_id,))
        if self._domain is not None:
            args.append("domain=%r" % (self._domain,))
        if self._metadata: # Stored as EmptyDict if None
            args.append("metadata=%r" % (self._metadata,))
        if self._domain_data is not None:
            args.append("domain_data=%r" % (self._domain_data,))

        return "%s(%s)" % (type(self).__name__, ', '.join(args))

    def __hash__(self):
        "Return a hash value for this Measure."
        hashdata = (self._domain_type, self._domain_id, hash(self._domain),
                    metadata_hashdata(self._metadata), id_or_none(self._domain_data))
        return hash(hashdata)

    def __eq__(self, other):
        "Checks if two Measures are equal."
        return (isinstance(other, Measure)
                and self._domain_type == other._domain_type
                and self._domain_id == other._domain_id
                and self._domain == other._domain
                and id_or_none(self._domain_data) == id_or_none(other._domain_data)
                and metadata_equal(self._metadata, other._metadata))

    def __add__(self, other):
        """Add two measures (self+other).

        Creates an intermediate object used for the notation

          expr * (dx(1) + dx(2)) := expr * dx(1) + expr * dx(2)
        """
        if isinstance(other, Measure):
            # Let dx(1) + dx(2) equal dx((1,2))
            return MeasureSum(self, other)
        else:
            # Can only add Measures
            return NotImplemented

    def __mul__(self, other):
        """Multiply two measures (self*other).

        Creates an intermediate object used for the notation

          expr * (dm1 * dm2) := expr * dm1 * dm2

        This is work in progress and not functional.
        """
        if isinstance(other, Measure):
            # Tensor product measure support
            return MeasureProduct(self, other)
        else:
            # Can't multiply Measure from the right with non-Measure type
            return NotImplemented

    def __rmul__(self, integrand):
        """Multiply a scalar expression with measure to construct a form with a single integral.

        This is to implement the notation

            form = integrand * self

        Integration properties are taken from this Measure object.
        """
        # Avoid circular imports
        from ufl.integral import Integral
        from ufl.form import Form

        # Allow python literals: 1*dx and 1.0*dx
        if isinstance(integrand, (int,float)):
            integrand = as_ufl(integrand)

        # Let other types implement multiplication with Measure
        # if they want to (to support the dolfin-adjoint TimeMeasure)
        if not isinstance(integrand, Expr):
            return NotImplemented

        # Allow only scalar integrands
        if not is_true_ufl_scalar(integrand):
            msg = ("Can only integrate scalar expressions. The integrand is a " +
                   "tensor expression with value rank %d and free indices %r.")
            error(msg % (integrand.rank(), integrand.free_indices()))

        # If we have a tuple of domain ids, delegate composition to Integral.__add__:
        domain_id = self.domain_id()
        if isinstance(domain_id, tuple):
            return sum(integrand*self.reconstruct(domain_id=d) for d in domain_id)

        # Check that we have an integer subdomain or a string
        # ("everywhere" or "otherwise", any more?)
        ufl_assert(isinstance(domain_id, (int,str)),
                   "Expecting integer or string domain id.")

        # If we don't have an integration domain, try to find one in integrand
        domain = self.domain()
        if domain is None:
            domains = integrand.domains()
            if len(domains) == 1:
                domain, = domains
            else:
                # TODO: Should this be an error? For now we need to support
                # assemble(1*dx, mesh=mesh) in dolfin for compatibility.
                # Maybe we can add a deprecation warning?
                #deprecation_warning("Integrals over undefined domains are deprecated.")
                domain = None

        # FIXME: Fix getitem so we can support this as well:
        # (probably need to store domain_data with Form or Integral?)
        # Suggestion to store canonically in Form:
        #   integral.domain_data() = value
        #   form.domain_data()[label][key] = value
        #   all(domain.data() == {} for domain in form.domains())
        # Then getitem data follows the data flow:
        #   dxs = dx[gd];  dxs._domain_data is gd
        #   dxs0 = dxs(0); dxs0._domain_data is gd
        #   M = 1*dxs0; M.integrals()[0].domain_data() is gd
        #   ; M.domain_data()[None][dxs.domain_type()] is gd
        #assemble(1*dx[cells] + 1*ds[bnd], mesh=mesh)

        # Otherwise create and return a one-integral form
        integral = Integral(integrand=integrand,
                            domain_type=self.domain_type(),
                            domain=domain,
                            domain_id=domain_id,
                            metadata=self.metadata(),
                            domain_data=self.domain_data())
        return Form([integral])

class MeasureSum(object):
    """Represents a sum of measures.

    This is a notational intermediate object to translate the notation

        f*(ds(1)+ds(3))

    into

        f*ds(1) + f*ds(3)
    """
    __slots__ = ("_measures",)
    def __init__(self, *measures):
        self._measures = measures

    def __rmul__(self, other):
        integrals = [other*m for m in self._measures]
        return sum(integrals)

    def __add__(self, other):
        if isinstance(other, Measure):
            return MeasureSum(*(self._measures + (other,)))
        elif isinstance(other, MeasureSum):
            return MeasureSum(*(self._measures + other._measures))
        return NotImplemented

    def __str__(self):
        return "{\n    " + "\n  + ".join(map(str,self._measures)) + "\n}"

class MeasureProduct(object):
    """Represents a product of measures.

    This is a notational intermediate object to handle the notation

        f*(dm1*dm2)

    This is work in progress and not functional. It needs support
    in other parts of ufl and the rest of the code generation chain.
    """
    __slots__ = ("_measures",)
    def __init__(self, *measures):
        "Create MeasureProduct from given list of measures."
        self._measures = measures
        ufl_assert(len(self._measures) > 1, "Expecting at least two measures.")

    def __mul__(self, other):
        """Flatten multiplication of product measures.

        This is to ensure that (dm1*dm2)*dm3 is stored as a
        simple list (dm1,dm2,dm3) in a single MeasureProduct.
        """
        if isinstance(other, Measure):
            measures = self.sub_measures() + [other]
            return MeasureProduct(*measures)
        else:
            return NotImplemented

    def __rmul__(self, integrand):
        error("TODO: Implement MeasureProduct.__rmul__ to construct integral and form somehow.")

    def sub_measures(self):
        "Return submeasures."
        return self._measures
