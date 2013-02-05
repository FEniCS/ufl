"""The Integral class."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Last changed: 2011-08-25

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.constantvalue import is_true_ufl_scalar, is_python_scalar
from ufl.expr import Expr
from ufl.domains import DomainDescription, Domain, Region, extract_top_domains, extract_domains, as_domain


# TODO: Move these somewhere more suitable?
def is_globally_constant(expr):
    """Check if an expression is globally constant, which
    includes spatially independent constant coefficients that
    are not known before assembly time."""

    from ufl.algorithms.traversal import traverse_terminals
    from ufl.argument import Argument
    from ufl.coefficient import Coefficient

    for e in traverse_terminals(expr):
        if isinstance(e, Argument):
            return False
        if isinstance(e, Coefficient) and e.element().family() != "Real":
            return False
        if not e.is_cellwise_constant():
            return False

    # All terminals passed constant check
    return True

# TODO: Move these somewhere more suitable?
def is_scalar_constant_expression(expr):
    """Check if an expression is a globally constant scalar expression."""
    if is_python_scalar(expr):
        return True
    if expr.shape() != ():
        return False
    return is_globally_constant(expr)


# TODO: Define some defaults as to how metadata should represent integration data here?
# quadrature_degree, quadrature_rule, ...

def register_domain_type(domain_type, measure_name):
    domain_type = domain_type.replace(" ", "_")
    if domain_type in Measure._domain_types:
        ufl_assert(Measure._domain_types[domain_type] == measure_name,
                   "Domain type already added with different measure name!")
        # Nothing to do
    else:
        ufl_assert(measure_name not in Measure._domain_types.values(),
                   "Measure name already used for another domain type!")
        Measure._domain_types[domain_type] = measure_name

class MeasureSum(object):
    """Notational intermediate object to translate the notation
    'f*(ds(1)+ds(3))'
    into
    'f*ds(1) + f*ds(3)'.
    Note that MeasureSum objects will never actually be part of forms.
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

def as_domain_type(domain_type):
    _domain_type = domain_type.replace(" ", "_")
    # Map short domain type name to long automatically
    if not _domain_type in Measure._domain_types:
        for k, v in Measure._domain_types.iteritems():
            if v == domain_type:
                _domain_type = k
                break
        # In the end, did we find a valid domain type?
        if not _domain_type in Measure._domain_types:
            error("Invalid domain type %s." % domain_type)
    return _domain_type

class Measure(object):
    """A measure for integration."""
    __slots__ = ("_domain_type",
                 "_domain_description",
                 "_metadata",
                 "_domain_data",
                 "_repr",)

    # Enumeration of special domain ids
    DOMAIN_ID_UNDEFINED       = "undefined"
    DOMAIN_ID_UNIQUE          = "unique"
    DOMAIN_ID_EVERYWHERE      = "everywhere"
    DOMAIN_ID_OTHERWISE       = "otherwise"
    DOMAIN_ID_DEFAULT   = DOMAIN_ID_EVERYWHERE # The one used by dx,ds,dS,etc.
    #DOMAIN_ID_DEFAULT   = DOMAIN_ID_UNIQUE # The one used by dx,ds,dS,etc.
    DOMAIN_ID_CONSTANTS = (DOMAIN_ID_UNDEFINED,
                           DOMAIN_ID_UNIQUE,
                           DOMAIN_ID_EVERYWHERE,
                           DOMAIN_ID_OTHERWISE,
                           )

    # Enumeration of valid domain types
    CELL           = "cell"
    EXTERIOR_FACET = "exterior_facet"
    INTERIOR_FACET = "interior_facet"
    POINT          = "point"
    MACRO_CELL     = "macro_cell"
    SURFACE        = "surface"
    _domain_types = { \
        CELL: "dx",
        EXTERIOR_FACET: "ds",
        INTERIOR_FACET: "dS",
        POINT: "dP",
        MACRO_CELL: "dE",
        SURFACE: "dc"
        }
    _domain_types_tuple = (CELL, EXTERIOR_FACET, INTERIOR_FACET,
                           POINT, MACRO_CELL, SURFACE)

    def __init__(self, domain_type, domain_id=None, metadata=None, domain_data=None):
        # Allow long domain type names with ' ' or '_'
        self._domain_type = as_domain_type(domain_type)

        # Can't use this constant as default value
        if domain_id is None:
            self._domain_description = Measure.DOMAIN_ID_DEFAULT
        else:
            self._domain_description = domain_id

        # Data for form compiler
        self._metadata = metadata

        # Data for problem solving environment
        self._domain_data = domain_data

        # TODO: Is repr of measure used anywhere anymore? Maybe we can fix this now:
        # NB! Deliberately ignoring domain data in repr, since it causes a bug
        # in the cache mechanisms in PyDOLFIN. I don't believe this is the
        # last word in this case, but it should fix all known problems for now.
        self._repr = "Measure(%r, %r, %r)" % (self._domain_type, self._domain_description,
                                              self._metadata)
        #self._repr = "Measure(%r, %r, %r, %r)" % (self._domain_type, self._domain_description,
        #                                          self._metadata, self._domain_data)

    def reconstruct(self, domain_id=None, metadata=None, domain_data=None):
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
            domain_id = self._domain_description
        if metadata is None:
            metadata = self._metadata
        if domain_data is None:
            domain_data = self._domain_data
        return Measure(self._domain_type, domain_id, metadata, domain_data)

    def domain_type(self):
        'Return the domain type, one of "cell", "exterior_facet", "interior_facet", etc.'
        return self._domain_type

    def domain_description(self):
        """Return the domain description of this measure.

        NB! Can be one of many types, this is work in progress!"""
        return self._domain_description

    def domain_id(self):
        "Return the domain id of this measure (integer)."
        #ufl_assert(isinstance(self._domain_description, int), "Measure does not have an integer domain id.") # TODO: Enable this
        return self._domain_description

    def metadata(self):
        """Return the integral metadata. This data is not interpreted by UFL.
        It is passed to the form compiler which can ignore it or use it to
        compile each integral of a form in a different way."""
        return self._metadata

    def domain_data(self):
        """Return the integral domain_data. This data is not interpreted by UFL.
        Its intension is to give a context in which the domain id is interpreted."""
        return self._domain_data

    # TODO: Figure out if we should keep this / how long to keep it / how to deprecate
    def __getitem__(self, domain_data):
        """Return a new Measure for same integration type with an attached
        context for interpreting domain ids. The default ID of this new Measure
        is undefined, and thus it must be qualified with a domain id to use
        in an integral. Example: dx = dx[boundaries]; L = f*v*dx(0) + g*v+dx(1)."""
        return self.reconstruct(domain_id=Measure.DOMAIN_ID_UNDEFINED,
                                domain_data=domain_data)

    def __call__(self, domain_id=None, metadata=None):
        """Return integral of same type on given sub domain,
        optionally with some metadata attached."""
        if domain_id is None and metadata is None:
            # Let syntax dx() mean integral over everywhere
            return self.reconstruct(domain_id=Measure.DOMAIN_ID_EVERYWHERE)
        else:
            # If we get keyword metadata, attatch it, if we get a domain id, use it
            return self.reconstruct(domain_id=domain_id, metadata=metadata)

    def __add__(self, other):
        if isinstance(other, Measure):
            # Let dx(1) + dx(2) equal dx((1,2))
            return MeasureSum(self, other)
        else:
            # Can only add Measures
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Measure):
            # Tensor product measure support
            return ProductMeasure(self, other)
        else:
            # Can't multiply Measure from the right with non-Measure type
            return NotImplemented

    def __rmul__(self, integrand):
        # Let other types implement multiplication with Measure
        # if they want to (to support the dolfin-adjoint TimeMeasure)
        if not isinstance(integrand, Expr):
            return NotImplemented

        # Allow only scalar integrands
        if not is_true_ufl_scalar(integrand):
            error("Trying to integrate expression of rank %d with free indices %r." \
                  % (integrand.rank(), integrand.free_indices()))

        # Is the measure in a state where multiplication is not allowed?
        if self._domain_description == Measure.DOMAIN_ID_UNDEFINED:
            error("Missing domain id. You need to select a subdomain, " +\
                  "e.g. M = f*dx(0) for subdomain 0.")

        #else: # TODO: Do it this way instead, and move all logic below into preprocess:
        #    # Return a one-integral form:
        #    from ufl.form import Form
        #    return Form( [Integral(integrand, self.domain_type(), self.domain_id(), self.metadata(), self.domain_data())] )
        #    # or if we move domain data into Form instead:
        #    integrals = [Integral(integrand, self.domain_type(), self.domain_id(), self.metadata())]
        #    domain_data = { self.domain_type(): self.domain_data() }
        #    return Form(integrals, domain_data)

        # TODO: How to represent all kinds of domain descriptions is still a bit unclear
        # Create form if we have a sufficient domain description
        elif (# We have a complete measure with domain description
            isinstance(self._domain_description, DomainDescription)
            # Is the measure in a basic state 'foo*dx'?
            or self._domain_description == Measure.DOMAIN_ID_UNIQUE
            # Is the measure over everywhere?
            or self._domain_description == Measure.DOMAIN_ID_EVERYWHERE
            # Is the measure in a state not allowed prior to preprocessing?
            or self._domain_description == Measure.DOMAIN_ID_OTHERWISE
            ):
            # Create and return a one-integral form
            from ufl.form import Form
            return Form( [Integral(integrand, self.domain_type(), self.domain_id(), self.metadata(), self.domain_data())] )

        # Did we get several ids?
        elif isinstance(self._domain_description, tuple):
            # FIXME: Leave this analysis to preprocessing
            return sum(integrand*self.reconstruct(domain_id=d) for d in self._domain_description)

        # Did we get a name?
        elif isinstance(self._domain_description, str):
            # FIXME: Leave this analysis to preprocessing

            # Get all domains and regions from integrand to analyse
            domains = extract_domains(integrand)

            # Get domain or region with this name from integrand, error if multiple found
            name = self._domain_description
            candidates = set()
            for TD in domains:
                if TD.name() == name:
                    candidates.add(TD)
            ufl_assert(len(candidates) > 0,
                       "Found no domain with name '%s' in integrand." % name)
            ufl_assert(len(candidates) == 1,
                       "Multiple distinct domains with same name encountered in integrand.")
            D, = candidates

            # Reconstruct measure with the found named domain or region
            measure = self.reconstruct(domain_id=D)
            return integrand*measure

        # Did we get a number?
        elif isinstance(self._domain_description, int):
            # FIXME: Leave this analysis to preprocessing

            # Get all top level domains from integrand to analyse
            domains = extract_top_domains(integrand)

            # Get domain from integrand, error if multiple found
            if len(domains) == 0:
                # This is the partially defined integral from dolfin expression mess
                cell = integrand.cell()
                D = None if cell is None else as_domain(cell)
            elif len(domains) == 1:
                D, = domains
            else:
                error("Ambiguous reference to integer subdomain with multiple top domains in integrand.")

            if D is None:
                # We have a number but not a domain? Leave it to preprocess...
                # This is the case with badly formed forms which can occur from dolfin
                # Create and return a one-integral form
                from ufl.form import Form
                return Form( [Integral(integrand, self.domain_type(), self.domain_id(), self.metadata(), self.domain_data())] )
            else:
                # Reconstruct measure with the found numbered subdomain
                measure = self.reconstruct(domain_id=D[self._domain_description])
                return integrand*measure

        # Provide error to user
        else:
            error("Invalid domain id %s." % str(self._domain_description))

    def __str__(self):
        d = Measure._domain_types[self._domain_type]
        metastring = "" if self._metadata is None else ("<%s>" % repr(self._metadata))
        return "%s%s%s" % (d, self._domain_description, metastring)

    def __repr__(self):
        return self._repr

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

class ProductMeasure(Measure):
    "Representation of a product measure."
    __slots__ = ("_measures",)
    def __init__(self, *measures):
        "Create ProductMeasure from given list of measures."
        self._measures = list(measures)
        ufl_assert(len(self._measures) > 0, "Expecting at least one measure")

        # FIXME: MER: The below is clearly wrong, but preprocess pose some
        # pretty hard restrictions. To be dealt with later.
        # self._domain_type = tuple(m.domain_type() for m in self._measures)
        # self._domain_description = tuple(m.domain_id() for m in self._measures)
        self._domain_type = measures[0].domain_type()
        self._domain_description = measures[0].domain_id()
        self._metadata = None
        self._domain_data = None
        self._repr = "ProductMeasure(*%r)" % self._measures

    def __mul__(self, other):
        "Flatten multiplication of product measures."
        if isinstance(other, Measure):
            measures = self.sub_measures() + [other]
            return ProductMeasure(*measures)
        #error("Can't multiply ProductMeasure from the right (with %r)." % (other,))
        return NotImplemented

    def __rmul__(self, integrand):
        error("TODO: Implement ProductMeasure.__rmul__ to construct integral and form somehow.")

    def sub_measures(self):
        "Return submeasures."
        return self._measures

# Transitional helper until we decide to change Integral permanently
def Integral2(integrand, domain_type, domain_desc, compiler_data, domain_data):
    return Integral(integrand, domain_type, domain_desc, compiler_data, domain_data)

class Integral(object):
    "An integral over a single domain."
    __slots__ = ("_integrand",
                 "_domain_type",
                 "_domain_description",
                 "_compiler_data",
                 "_domain_data", # TODO: Make this part of Form instead?
                 "_measure", # TODO: Remove when not used anywhere
                 )
    def __init__(self, integrand, domain_type, domain_description, compiler_data, domain_data):
        ufl_assert(isinstance(integrand, Expr),
                   "Expecting integrand to be an Expr instance.")
        self._integrand = integrand

        self._domain_type = domain_type
        self._domain_description = domain_description
        self._compiler_data = compiler_data
        self._domain_data = domain_data

        # TODO: Remove this, kept for a transitional period:
        self._measure = Measure(domain_type, domain_description, compiler_data, domain_data)

    def reconstruct(self, integrand=None, domain_type=None, domain_description=None, compiler_data=None, domain_data=None):
        """Construct a new Integral object with some properties replaced with new values.

        Example:
            <a = Integral instance>
            b = a.reconstruct(expand_compounds(a.integrand()))
            c = a.reconstruct(compiler_data={'quadrature_degree':2})
        """
        if integrand is None:
            integrand = self.integrand()
        if domain_type is None:
            domain_type = self.domain_type()
        if domain_description is None:
            domain_description = self.domain_description()
        if compiler_data is None:
            compiler_data = self.compiler_data()
        if domain_data is None:
            domain_data = self.domain_data()
        return Integral(integrand, domain_type, domain_description, compiler_data, domain_data)

    def integrand(self):
        "Return the integrand expression, which is an Expr instance."
        return self._integrand

    def domain_type(self):
        "Return the domain type of this integral."
        return self._domain_type

    def domain_description(self):
        """Return the domain description of this integral.

        NB! Can be one of many types, this is work in progress!"""
        return self._domain_description

    def domain_id(self):
        "Return the domain id of this integral."
        #ufl_assert(isinstance(self._domain_description, int), "Integral does not have an integer domain id.") # TODO: Enable this
        return self._domain_description

    def compiler_data(self):
        "Return the compiler metadata this integral has been annotated with."
        return self._compiler_data

    def domain_data(self): # TODO: Move to Form?
        "Return the assembler metadata this integral has been annotated with."
        return self._domain_data

    def measure(self): # TODO: Remove this
        "Return the measure associated with this integral."
        return self._measure

    def __neg__(self):
        return self.reconstruct(-self._integrand)

    def __mul__(self, scalar):
        ufl_assert(is_python_scalar(scalar),
                   "Cannot multiply an integral with non-constant values.")
        return self.reconstruct(scalar*self._integrand)

    def __rmul__(self, scalar):
        ufl_assert(is_scalar_constant_expression(scalar),
                   "An integral can only be multiplied by a "
                   "globally constant scalar expression.")
        return self.reconstruct(scalar*self._integrand)

    def __str__(self):
        return "{ %s } * %s" % (self._integrand, self._measure) # FIXME

    def __repr__(self):
        return "Integral(%r, %r, %r, %r, %r)" % (self._integrand, self._domain_type, self._domain_description,
                                                 self._compiler_data, self._domain_data)

    def __eq__(self, other):
        return (self._domain_type == other._domain_type
                and self._domain_description == other._domain_description
                and self._integrand == other._integrand
                and self._compiler_data == other._compiler_data
                and self._domain_data == other._domain_data
                )

    def __hash__(self):
        data = (hash(self._integrand), self._domain_type, self._domain_description)
        # Assuming we can get away with few collisions by ignoring metadata:
        # hash(self._compiler_data), hash(self._domain_data)
        return hash(data)
