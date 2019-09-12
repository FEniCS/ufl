# -*- coding: utf-8 -*-
"""The Measure class."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg 2008-2016
# Modified by Massimiliano Leoni, 2016.

import numbers

from ufl.utils.str import as_native_strings
from ufl.utils.str import as_native_str
from ufl.log import error, deprecate
from ufl.core.expr import Expr
from ufl.checks import is_true_ufl_scalar
from ufl.constantvalue import as_ufl
from ufl.utils.dicts import EmptyDict
from ufl.domain import as_domain, AbstractDomain, extract_domains
from ufl.protocols import id_or_none, metadata_equal, metadata_hashdata


# Export list for ufl.classes
__all_classes__ = as_native_strings(["Measure", "MeasureSum", "MeasureProduct"])


# TODO: Design a class IntegralType(name, shortname, codim, num_cells, ...)?
# TODO: Improve descriptions below:

# Enumeration of valid domain types
_integral_types = [
    # === Integration over full topological dimension:
    ("cell", "dx"),                     # Over cells of a mesh
    # ("macro_cell", "dE"),              # Over a group of adjacent cells (TODO: Arbitrary cell group? Not currently used.)

    # === Integration over topological dimension - 1:
    ("exterior_facet", "ds"),           # Over one-sided exterior facets of a mesh
    ("interior_facet", "dS"),           # Over two-sided facets between pairs of adjacent cells of a mesh

    # === Integration over topological dimension 0
    ("vertex", "dP"),                    # Over vertices of a mesh
    # ("vertex", "dV"),                  # TODO: Use this over vertices?
    # ("point", "dP"),                   # TODO: Use this over arbitrary points inside cells?

    # === Integration over custom domains
    ("custom", "dc"),                 # Over custom user-defined domains (run-time quadrature points)
    ("cutcell", "dC"),                # Over a cell with some part cut away (run-time quadrature points)
    ("interface", "dI"),              # Over a facet fragment overlapping with two or more cells (run-time quadrature points)
    ("overlap", "dO"),                # Over a cell fragment overlapping with two or more cells (run-time quadrature points)

    # === Firedrake specific hacks on the way out:
    # TODO: Remove these, firedrake can use metadata instead
    # and create the measure objects in firedrake:
    ("exterior_facet_bottom", "ds_b"),  # Over bottom facets on extruded mesh
    ("exterior_facet_top", "ds_t"),     # Over top facets on extruded mesh
    ("exterior_facet_vert", "ds_v"),    # Over side facets of an extruded mesh
    ("interior_facet_horiz", "dS_h"),   # Over horizontal facets of an extruded mesh
    ("interior_facet_vert", "dS_v"),  # Over vertical facets of an extruded mesh
]

integral_type_to_measure_name = dict((l, s) for l, s in _integral_types)
measure_name_to_integral_type = dict((s, l) for l, s in _integral_types)

custom_integral_types = ("custom", "cutcell", "interface", "overlap")
point_integral_types = ("vertex",)  # "point")
facet_integral_types = ("exterior_facet", "interior_facet")


def register_integral_type(integral_type, measure_name):
    global integral_type_to_measure_name, measure_name_to_integral_type
    if measure_name != integral_type_to_measure_name.get(integral_type, measure_name):
        error("Integral type already added with different measure name!")
    if integral_type != measure_name_to_integral_type.get(measure_name, integral_type):
        error("Measure name already used for another domain type!")
    integral_type_to_measure_name[integral_type] = measure_name
    measure_name_to_integral_type[measure_name] = integral_type


def as_integral_type(integral_type):
    "Map short name to long name and require a valid one."
    integral_type = integral_type.replace(" ", "_")
    integral_type = measure_name_to_integral_type.get(integral_type,
                                                      integral_type)
    if integral_type not in integral_type_to_measure_name:
        error("Invalid integral_type.")
    return integral_type


def integral_types():
    "Return a tuple of all domain type strings."
    return tuple(sorted(integral_type_to_measure_name.keys()))


def measure_names():
    "Return a tuple of all measure name strings."
    return tuple(sorted(measure_name_to_integral_type.keys()))


class Measure(object):
    __slots__ = as_native_strings(("_integral_type",
                                   "_domain",
                                   "_subdomain_id",
                                   "_metadata",
                                   "_subdomain_data"))
    """Representation of an integration measure.

    The Measure object holds information about integration properties
    to be transferred to a Form on multiplication with a scalar
    expression.

    """

    def __init__(self,
                 integral_type,  # "dx" etc
                 domain=None,
                 subdomain_id="everywhere",
                 metadata=None,
                 subdomain_data=None):
        """
        integral_type:
            str, one of "cell", etc.,
            or short form "dx", etc.

        domain:
            an AbstractDomain object (most often a Mesh)

        subdomain_id:
            either string "everywhere",
            a single subdomain id int,
            or tuple of ints

        metadata:
            dict, with additional compiler-specific parameters
            affecting how code is generated, including parameters
            for optimization or debugging of generated code.

        subdomain_data:
            object representing data to interpret subdomain_id with.
        """
        # Map short name to long name and require a valid one
        self._integral_type = as_integral_type(integral_type)

        # Check that we either have a proper AbstractDomain or none
        self._domain = None if domain is None else as_domain(domain)
        if not (self._domain is None or isinstance(self._domain, AbstractDomain)):
            error("Invalid domain.")

        # Store subdomain data
        self._subdomain_data = subdomain_data
        # FIXME: Cannot require this (yet) because we currently have
        # no way to implement ufl_id for dolfin SubDomain
        # if not (self._subdomain_data is None or hasattr(self._subdomain_data, "ufl_id")):
        #     error("Invalid domain data, missing ufl_id() implementation.")

        # Accept "everywhere", single subdomain, or multiple
        # subdomains
        if isinstance(subdomain_id, tuple):
            for did in subdomain_id:
                if not isinstance(did, numbers.Integral):
                    error("Invalid subdomain_id %s." % (did,))
        else:
            if not (subdomain_id in ("everywhere",) or isinstance(subdomain_id, numbers.Integral)):
                error("Invalid subdomain_id %s." % (subdomain_id,))
        self._subdomain_id = subdomain_id

        # Validate compiler options are None or dict
        if metadata is not None and not isinstance(metadata, dict):
            error("Invalid metadata.")
        self._metadata = metadata or EmptyDict

    def integral_type(self):
        """Return the domain type.

        Valid domain types are "cell", "exterior_facet", "interior_facet", etc.
        """
        return self._integral_type

    def ufl_domain(self):
        """Return the domain associated with this measure.

        This may be None or a Domain object.
        """
        return self._domain

    def subdomain_id(self):
        "Return the domain id of this measure (integer)."
        return self._subdomain_id

    def metadata(self):
        """Return the integral metadata. This data is not interpreted by UFL.
        It is passed to the form compiler which can ignore it or use
        it to compile each integral of a form in a different way.

        """
        return self._metadata

    def reconstruct(self,
                    integral_type=None,
                    subdomain_id=None,
                    domain=None,
                    metadata=None,
                    subdomain_data=None):
        """Construct a new Measure object with some properties replaced with
        new values.

        Example:
        -------
            <dm = Measure instance>
            b = dm.reconstruct(subdomain_id=2)
            c = dm.reconstruct(metadata={ "quadrature_degree": 3 })

        Used by the call operator, so this is equivalent:
            b = dm(2)
            c = dm(0, { "quadrature_degree": 3 })

        """
        if subdomain_id is None:
            subdomain_id = self.subdomain_id()
        if domain is None:
            domain = self.ufl_domain()
        if metadata is None:
            metadata = self.metadata()
        if subdomain_data is None:
            subdomain_data = self.subdomain_data()
        return Measure(self.integral_type(),
                       domain=domain, subdomain_id=subdomain_id,
                       metadata=metadata, subdomain_data=subdomain_data)

    def subdomain_data(self):
        """Return the integral subdomain_data. This data is not interpreted by
        UFL.  Its intension is to give a context in which the domain
        id is interpreted.

        """
        return self._subdomain_data

    # Note: Must keep the order of the first two arguments here
    # (subdomain_id, metadata) for backwards compatibility, because
    # some tutorials write e.g. dx(0, {...}) to set metadata.
    def __call__(self, subdomain_id=None, metadata=None, domain=None,
                 subdomain_data=None, degree=None, scheme=None, rule=None):
        """Reconfigure measure with new domain specification or metadata."""

        # Deprecation of 'rule' in favour of 'scheme'
        if rule is not None:
            deprecate("Measure argument 'rule' has been renamed to 'scheme'.")
            assert scheme is None or scheme == rule
            scheme = rule

        # Let syntax dx() mean integral over everywhere
        all_args = (subdomain_id, metadata, domain, subdomain_data,
                    degree, scheme)
        if all(arg is None for arg in all_args):
            return self.reconstruct(subdomain_id="everywhere")

        # Let syntax dx(domain) or dx(domain, metadata) mean integral
        # over entire domain.  To do this we need to hijack the first
        # argument:
        if subdomain_id is not None and (isinstance(subdomain_id,
                                                    AbstractDomain) or
                                         hasattr(subdomain_id, 'ufl_domain')):
            if domain is not None:
                error("Ambiguous: setting domain both as keyword argument and first argument.")
            subdomain_id, domain = "everywhere", as_domain(subdomain_id)

        # If degree or scheme is set, inject into metadata. This is a
        # quick fix to enable the dx(..., degree=3) notation.
        # TODO: Make degree and scheme properties of integrals instead of adding to metadata.
        if (degree, scheme) != (None, None):
            metadata = {} if metadata is None else metadata.copy()
            if degree is not None:
                metadata["quadrature_degree"] = degree
            if scheme is not None:
                metadata["quadrature_rule"] = scheme

        # If we get any keywords, use them to reconstruct Measure.
        # Note that if only one argument is given, it is the
        # subdomain_id, e.g. dx(3) == dx(subdomain_id=3)
        return self.reconstruct(subdomain_id=subdomain_id, domain=domain,
                                metadata=metadata,
                                subdomain_data=subdomain_data)

    def __getitem__(self, data):
        """This operator supports legacy syntax in python dolfin programs.

        The old documentation reads: Return a new Measure for same
        integration type with an attached context for interpreting
        domain ids. By default this new Measure integrates over
        everywhere, but it can be restricted with a domain id as
        usual. Example: dx = dx[boundaries]; L = f*v*dx + g*v+dx(1).

        """
        deprecate("Notation dx[meshfunction] is deprecated. Please use dx(subdomain_data=meshfunction) instead.")
        return self(subdomain_data=data)

    def __str__(self):
        global integral_type_to_measure_name
        name = integral_type_to_measure_name[self._integral_type]
        args = []

        if self._subdomain_id is not None:
            args.append("subdomain_id=%s" % (self._subdomain_id,))
        if self._domain is not None:
            args.append("domain=%s" % (self._domain,))
        if self._metadata:  # Stored as EmptyDict if None
            args.append("metadata=%s" % (self._metadata,))
        if self._subdomain_data is not None:
            args.append("subdomain_data=%s" % (self._subdomain_data,))

        return "%s(%s)" % (name, ', '.join(args))

    def __repr__(self):
        "Return a repr string for this Measure."
        global integral_type_to_measure_name

        args = []
        args.append(repr(self._integral_type))

        if self._subdomain_id is not None:
            args.append("subdomain_id=%s" % repr(self._subdomain_id))
        if self._domain is not None:
            args.append("domain=%s" % repr(self._domain))
        if self._metadata:  # Stored as EmptyDict if None
            args.append("metadata=%s" % repr(self._metadata))
        if self._subdomain_data is not None:
            args.append("subdomain_data=%s" % repr(self._subdomain_data))

        r = "%s(%s)" % (type(self).__name__, ', '.join(args))
        return as_native_str(r)

    def __hash__(self):
        "Return a hash value for this Measure."
        hashdata = (self._integral_type,
                    self._subdomain_id,
                    hash(self._domain),
                    metadata_hashdata(self._metadata),
                    id_or_none(self._subdomain_data))
        return hash(hashdata)

    def __eq__(self, other):
        "Checks if two Measures are equal."
        return (isinstance(other, Measure) and
                self._integral_type == other._integral_type and
                self._subdomain_id == other._subdomain_id and
                self._domain == other._domain and
                id_or_none(self._subdomain_data) == id_or_none(other._subdomain_data) and
                metadata_equal(self._metadata, other._metadata))

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
        """Multiply a scalar expression with measure to construct a form with
        a single integral.

        This is to implement the notation

            form = integrand * self

        Integration properties are taken from this Measure object.

        """
        # Avoid circular imports
        from ufl.integral import Integral
        from ufl.form import Form

        # Allow python literals: 1*dx and 1.0*dx
        if isinstance(integrand, (int, float)):
            integrand = as_ufl(integrand)

        # Let other types implement multiplication with Measure if
        # they want to (to support the dolfin-adjoint TimeMeasure)
        if not isinstance(integrand, Expr):
            return NotImplemented

        # Allow only scalar integrands
        if not is_true_ufl_scalar(integrand):
            error("Can only integrate scalar expressions. The integrand is a "
                  "tensor expression with value shape %s and free indices with labels %s." %
                  (integrand.ufl_shape, integrand.ufl_free_indices))

        # If we have a tuple of domain ids, delegate composition to
        # Integral.__add__:
        subdomain_id = self.subdomain_id()
        if isinstance(subdomain_id, tuple):
            return sum(integrand * self.reconstruct(subdomain_id=d) for d in subdomain_id)

        # Check that we have an integer subdomain or a string
        # ("everywhere" or "otherwise", any more?)
        if not isinstance(subdomain_id, (str, numbers.Integral,)):
            error("Expecting integer or string domain id.")

        # If we don't have an integration domain, try to find one in
        # integrand
        domain = self.ufl_domain()
        if domain is None:
            domains = extract_domains(integrand)
            if len(domains) == 1:
                domain, = domains
            elif len(domains) == 0:
                error("This integral is missing an integration domain.")
            else:
                error("Multiple domains found, making the choice of integration domain ambiguous.")

        # Otherwise create and return a one-integral form
        integral = Integral(integrand=integrand,
                            integral_type=self.integral_type(),
                            domain=domain,
                            subdomain_id=subdomain_id,
                            metadata=self.metadata(),
                            subdomain_data=self.subdomain_data())
        return Form([integral])


class MeasureSum(object):
    """Represents a sum of measures.

    This is a notational intermediate object to translate the notation

        f*(ds(1)+ds(3))

    into

        f*ds(1) + f*ds(3)
    """
    __slots__ = as_native_strings(("_measures",))

    def __init__(self, *measures):
        self._measures = measures

    def __rmul__(self, other):
        integrals = [other * m for m in self._measures]
        return sum(integrals)

    def __add__(self, other):
        if isinstance(other, Measure):
            return MeasureSum(*(self._measures + (other,)))
        elif isinstance(other, MeasureSum):
            return MeasureSum(*(self._measures + other._measures))
        return NotImplemented

    def __str__(self):
        return "{\n    " + "\n  + ".join(map(str, self._measures)) + "\n}"


class MeasureProduct(object):
    """Represents a product of measures.

    This is a notational intermediate object to handle the notation

        f*(dm1*dm2)

    This is work in progress and not functional. It needs support
    in other parts of ufl and the rest of the code generation chain.

    """
    __slots__ = as_native_strings(("_measures",))

    def __init__(self, *measures):
        "Create MeasureProduct from given list of measures."
        self._measures = measures
        if len(self._measures) < 2:
            error("Expecting at least two measures.")

    def __mul__(self, other):
        """Flatten multiplication of product measures.

        This is to ensure that (dm1*dm2)*dm3 is stored as a simple
        list (dm1,dm2,dm3) in a single MeasureProduct.

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
