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
from itertools import chain

from ufl.checks import is_true_ufl_scalar
from ufl.constantvalue import as_ufl
from ufl.core.expr import Expr
from ufl.domain import AbstractDomain, as_domain, extract_domains
from ufl.protocols import id_or_none

# Export list for ufl.classes
__all_classes__ = ["Measure", "MeasureSum", "MeasureProduct"]


# TODO: Design a class IntegralType(name, shortname, codim, num_cells, ...)?
# TODO: Improve descriptions below:

# Enumeration of valid domain types
_integral_types = [
    # === Integration over full topological dimension:
    ("cell", "dx"),  # Over cells of a mesh
    # === Integration over topological dimension - 1:
    ("exterior_facet", "ds"),  # Over one-sided exterior facets of a mesh
    ("interior_facet", "dS"),  # Over two-sided facets between pairs of adjacent cells of a mesh
    # === Integration over topological dimension - 2:
    ("ridge", "dr"),  # Over ridges of a mesh
    # === Integration over topological dimension 0
    ("vertex", "dP"),  # Over vertices of a mesh
    # === Integration over custom domains
    ("custom", "dc"),  # Over custom user-defined domains (run-time quadrature points)
    ("cutcell", "dC"),  # Over a cell with some part cut away (run-time quadrature points)
    (
        "interface",
        "dI",
    ),  # Over a facet fragment overlapping with two or more cells (run-time quadrature points)
    (
        "overlap",
        "dO",
    ),  # Over a cell fragment overlapping with two or more cells (run-time quadrature points)
    # === Firedrake specifics:
    ("exterior_facet_bottom", "ds_b"),  # Over bottom facets on extruded mesh
    ("exterior_facet_top", "ds_t"),  # Over top facets on extruded mesh
    ("exterior_facet_vert", "ds_v"),  # Over side facets of an extruded mesh
    ("interior_facet_horiz", "dS_h"),  # Over horizontal facets of an extruded mesh
    ("interior_facet_vert", "dS_v"),  # Over vertical facets of an extruded mesh
]

integral_type_to_measure_name = {i: s for i, s in _integral_types}
measure_name_to_integral_type = {s: i for i, s in _integral_types}

custom_integral_types = ("custom", "cutcell", "interface", "overlap")
point_integral_types = ("vertex",)  # "point")
facet_integral_types = ("exterior_facet", "interior_facet")
ridge_integral_types = ("ridge",)


def register_integral_type(integral_type, measure_name):
    """Register an integral type."""
    global integral_type_to_measure_name, measure_name_to_integral_type
    if measure_name != integral_type_to_measure_name.get(integral_type, measure_name):
        raise ValueError("Integral type already added with different measure name!")
    if integral_type != measure_name_to_integral_type.get(measure_name, integral_type):
        raise ValueError("Measure name already used for another domain type!")
    integral_type_to_measure_name[integral_type] = measure_name
    measure_name_to_integral_type[measure_name] = integral_type


def as_integral_type(integral_type):
    """Map short name to long name and require a valid one."""
    integral_type = integral_type.replace(" ", "_")
    integral_type = measure_name_to_integral_type.get(integral_type, integral_type)
    if integral_type not in integral_type_to_measure_name:
        raise ValueError(f"Invalid integral_type: {integral_type}.")
    return integral_type


def integral_types():
    """Return a tuple of all domain type strings."""
    return tuple(sorted(integral_type_to_measure_name.keys()))


def measure_names():
    """Return a tuple of all measure name strings."""
    return tuple(sorted(measure_name_to_integral_type.keys()))


class Measure:
    """Representation of an integration measure.

    The Measure object holds information about integration properties
    to be transferred to a Form on multiplication with a scalar
    expression.
    """

    __slots__ = ("_domain", "_integral_type", "_metadata", "_subdomain_data", "_subdomain_id")

    def __init__(
        self,
        integral_type,  # "dx" etc
        domain=None,
        subdomain_id="everywhere",
        metadata=None,
        subdomain_data=None,
    ):
        """Initialise.

        Args:
            integral_type: one of "cell", etc, or short form "dx", etc
            domain: an AbstractDomain object (most often a Mesh)
            subdomain_id: either string "everywhere", a single subdomain id int, or tuple of ints
            metadata: dict, with additional compiler-specific parameters
                affecting how code is generated, including parameters
                for optimization or debugging of generated code
            subdomain_data: object representing data to interpret subdomain_id with
        """
        # Map short name to long name and require a valid one
        self._integral_type = as_integral_type(integral_type)

        # Check that we either have a proper AbstractDomain or none
        if domain is not None:
            domain = as_domain(domain)
            if not isinstance(domain, AbstractDomain):
                raise ValueError("Invalid domain.")
        self._domain = domain

        # Store subdomain data
        self._subdomain_data = subdomain_data
        # FIXME: Cannot require this (yet) because we currently have
        # no way to implement ufl_id for dolfin SubDomain
        # if not (self._subdomain_data is None or hasattr(self._subdomain_data, "ufl_id")):
        #     raise ValueError("Invalid domain data, missing ufl_id() implementation.")

        # Accept "everywhere", single subdomain, or multiple
        # subdomains
        if isinstance(subdomain_id, tuple):
            for did in subdomain_id:
                if not isinstance(did, numbers.Integral):
                    raise ValueError(f"Invalid subdomain_id {did}.")
        else:
            if not (subdomain_id in ("everywhere",) or isinstance(subdomain_id, numbers.Integral)):
                raise ValueError(f"Invalid subdomain_id {subdomain_id}.")
        self._subdomain_id = subdomain_id

        # Validate compiler options are None or dict
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Invalid metadata.")
        self._metadata = metadata or {}

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
        """Return the domain id of this measure (integer)."""
        return self._subdomain_id

    def metadata(self):
        """Return the integral metadata.

        This data is not interpreted by UFL.
        It is passed to the form compiler which can ignore it or use
        it to compile each integral of a form in a different way.
        """
        return self._metadata

    def reconstruct(
        self, integral_type=None, subdomain_id=None, domain=None, metadata=None, subdomain_data=None
    ):
        """Construct a new Measure object with some properties replaced with new values.

        Example:
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
        return Measure(
            self.integral_type(),
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=metadata,
            subdomain_data=subdomain_data,
        )

    def subdomain_data(self):
        """Return the integral subdomain_data.

        This data is not interpreted by
        UFL.  Its intension is to give a context in which the domain
        id is interpreted.
        """
        return self._subdomain_data

    # Note: Must keep the order of the first two arguments here
    # (subdomain_id, metadata) for backwards compatibility, because
    # some tutorials write e.g. dx(0, {...}) to set metadata.
    def __call__(
        self,
        subdomain_id=None,
        metadata=None,
        domain=None,
        subdomain_data=None,
        degree=None,
        scheme=None,
    ):
        """Reconfigure measure with new domain specification or metadata."""
        # Let syntax dx() mean integral over everywhere
        all_args = (subdomain_id, metadata, domain, subdomain_data, degree, scheme)
        if all(arg is None for arg in all_args):
            return self.reconstruct(subdomain_id="everywhere")

        # Let syntax dx(domain) or dx(domain, metadata) mean integral
        # over entire domain.  To do this we need to hijack the first
        # argument:
        if subdomain_id is not None and (
            isinstance(subdomain_id, AbstractDomain) or hasattr(subdomain_id, "ufl_domain")
        ):
            if domain is not None:
                raise ValueError(
                    "Ambiguous: setting domain both as keyword argument and first argument."
                )
            subdomain_id, domain = "everywhere", subdomain_id

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
        return self.reconstruct(
            subdomain_id=subdomain_id,
            domain=domain,
            metadata=metadata,
            subdomain_data=subdomain_data,
        )

    def __str__(self):
        """Format as a string."""
        name = integral_type_to_measure_name[self._integral_type]
        args = []

        if self._subdomain_id is not None:
            args.append(f"subdomain_id={self._subdomain_id}")
        if self._domain is not None:
            args.append(f"domain={self._domain}")
        if self._metadata:  # Stored as {} if None
            args.append(f"metadata={self._metadata}")
        if self._subdomain_data is not None:
            args.append(f"subdomain_data={self._subdomain_data}")

        return "{}({})".format(name, ", ".join(args))

    def __repr__(self):
        """Return a repr string for this Measure."""
        args = []
        args.append(repr(self._integral_type))

        if self._subdomain_id is not None:
            args.append(f"subdomain_id={self._subdomain_id!r}")
        if self._domain is not None:
            args.append(f"domain={self._domain!r}")
        if self._metadata:  # Stored as {} if None
            args.append(f"metadata={self._metadata!r}")
        if self._subdomain_data is not None:
            args.append(f"subdomain_data={self._subdomain_data!r}")

        r = "{}({})".format(type(self).__name__, ", ".join(args))
        return r

    def __hash__(self):
        """Return a hash value for this Measure."""
        metadata_hashdata = tuple(sorted((k, id(v)) for k, v in list(self._metadata.items())))
        hashdata = (
            self._integral_type,
            self._subdomain_id,
            hash(self._domain),
            metadata_hashdata,
            id_or_none(self._subdomain_data),
        )
        return hash(hashdata)

    def __eq__(self, other):
        """Checks if two Measures are equal."""
        if not isinstance(other, Measure):
            return False

        sorted_metadata = sorted((k, id(v)) for k, v in list(self._metadata.items()))
        sorted_other_metadata = sorted((k, id(v)) for k, v in list(other._metadata.items()))

        return (
            self._integral_type == other._integral_type
            and self._subdomain_id == other._subdomain_id
            and self._domain == other._domain
            and id_or_none(self._subdomain_data) == id_or_none(other._subdomain_data)
            and sorted_metadata == sorted_other_metadata
        )

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
        from ufl.form import Form
        from ufl.integral import Integral

        # Allow python literals: 1*dx and 1.0*dx
        if isinstance(integrand, (int, float)):
            integrand = as_ufl(integrand)

        # Let other types implement multiplication with Measure if
        # they want to (to support the dolfin-adjoint TimeMeasure)
        if not isinstance(integrand, Expr):
            return NotImplemented

        # Allow only scalar integrands
        if not is_true_ufl_scalar(integrand):
            raise ValueError(
                "Can only integrate scalar expressions. The integrand is a "
                f"tensor expression with value shape {integrand.ufl_shape} and "
                f"free indices with labels {integrand.ufl_free_indices}."
            )

        # If we have a tuple of domain ids build the integrals one by
        # one and construct as a Form in one go.
        subdomain_id = self.subdomain_id()
        if isinstance(subdomain_id, tuple):
            return Form(
                list(
                    chain(
                        *(
                            (integrand * self.reconstruct(subdomain_id=d)).integrals()
                            for d in subdomain_id
                        )
                    )
                )
            )

        # Check that we have an integer subdomain or a string
        # ("everywhere" or "otherwise", any more?)
        if not isinstance(
            subdomain_id,
            (
                str,
                numbers.Integral,
            ),
        ):
            raise ValueError("Expecting integer or string domain id.")

        # If we don't have an integration domain, try to find one in
        # integrand
        domain = self.ufl_domain()
        if domain is None:
            domains = extract_domains(integrand)
            if len(domains) == 1:
                (domain,) = domains
            elif len(domains) == 0:
                raise ValueError("This integral is missing an integration domain.")
            else:
                raise ValueError(
                    "Multiple domains found, making the choice of integration domain ambiguous."
                )

        # Otherwise create and return a one-integral form
        integral = Integral(
            integrand=integrand,
            integral_type=self.integral_type(),
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=self.metadata(),
            subdomain_data=self.subdomain_data(),
        )
        return Form([integral])


class MeasureSum:
    """Represents a sum of measures.

    This is a notational intermediate object to translate the notation
        f*(ds(1)+ds(3))
    into
        f*ds(1) + f*ds(3)
    """

    __slots__ = ("_measures",)

    def __init__(self, *measures):
        """Initialise."""
        self._measures = measures

    def __rmul__(self, other):
        """Multiply."""
        integrals = [other * m for m in self._measures]
        return sum(integrals)

    def __add__(self, other):
        """Add."""
        if isinstance(other, Measure):
            return MeasureSum(*(self._measures + (other,)))
        elif isinstance(other, MeasureSum):
            return MeasureSum(*(self._measures + other._measures))
        return NotImplemented

    def __str__(self):
        """Format as a string."""
        return "{\n    " + "\n  + ".join(map(str, self._measures)) + "\n}"


class MeasureProduct:
    """Represents a product of measures.

    This is a notational intermediate object to handle the notation

        f*(dm1*dm2)

    This is work in progress and not functional. It needs support
    in other parts of ufl and the rest of the code generation chain.
    """

    __slots__ = ("_measures",)

    def __init__(self, *measures):
        """Create MeasureProduct from given list of measures."""
        self._measures = measures
        if len(self._measures) < 2:
            raise ValueError("Expecting at least two measures.")

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
        """Multiply."""
        # TODO: Implement MeasureProduct.__rmul__ to construct integral and form somehow.
        raise NotImplementedError()

    def sub_measures(self):
        """Return submeasures."""
        return self._measures
