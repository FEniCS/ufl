"""The Integral class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-13"

# Modified by Anders Logg, 2008

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.constantvalue import is_true_ufl_scalar

class Measure(object):
    """A measure for integration."""
    __slots__ = ("_domain_type", "_domain_id", "_metadata", "_repr")
    def __init__(self, domain_type, domain_id, metadata = None):
        self._domain_type = domain_type.replace(" ", "_")
        self._domain_id   = domain_id
        self._metadata    = metadata
        self._repr        = "Measure(%r, %r, %r)" % (self._domain_type, self._domain_id, self._metadata)
    
    def reconstruct(self, domain_id=None, metadata=None):
        """Construct a new Measure object with some properties replaced with new values.
        
        Example:
            <a = Measure instance>
            b = a.reconstruct(domain_id=2)
            c = a.reconstruct(metadata={"quadrature_order":3})
        
        Used by the call operator, so this is equivalent:
            b = a(2)
            c = a(0, {"quadrature_order":3})
        """
        if domain_id is None: domain_id = self._domain_id
        if metadata  is None: metadata  = self._metadata
        return Measure(self._domain_type, domain_id, metadata)
    
    # Enumeration of valid domain types
    CELL = "cell"
    EXTERIOR_FACET = "exterior_facet"
    INTERIOR_FACET = "interior_facet"
    
    def domain_type(self):
        'Return the domain type, one of "cell", "exterior_facet" or "interior_facet".'
        return self._domain_type
    
    def domain_id(self):
        "Return the domain id (integer)."
        return self._domain_id
    
    def metadata(self):
        "Return the integral metadata. What this can be is currently undefined." # TODO!
        return self._metadata
    
    def __call__(self, domain_id, metadata=None): # TODO: Define how metadata should represent integration data here: quadrature_degree, quadrature_rule, ...
        "Return integral of same type on given sub domain"
        return self.reconstruct(domain_id=domain_id, metadata=metadata)
    
    def __mul__(self, other):
        error("Can't multiply Measure from the right (with %r)." % (other,))
    
    def __rmul__(self, integrand):
        ufl_assert(is_true_ufl_scalar(integrand),   
            "Trying to integrate expression of rank %d with free indices %r." \
            % (integrand.rank(), integrand.free_indices()))
        from ufl.form import Form
        return Form( [Integral(integrand, self)] )
    
    def __str__(self):
        d = { Measure.CELL: "dx",
              Measure.EXTERIOR_FACET: "ds",
              Measure.INTERIOR_FACET: "dS"
            }[self._domain_type]
        metastring = "" if self._metadata is None else ("<%s>" % repr(self._metadata))
        return "%s%d%s" % (d, self._domain_id, metastring)
    
    def __repr__(self):
        return self._repr
    
    def __hash__(self):
        return hash(self._repr)
    
    def __eq__(self, other):
        return repr(self) == repr(other)

class Integral(object):
    "An integral over a single domain."
    __slots__ = ("_integrand", "_measure", "_repr")
    def __init__(self, integrand, measure):
        from ufl.expr import Expr
        ufl_assert(isinstance(integrand, Expr), "Expecting integrand to be an Expr instance.")
        ufl_assert(isinstance(measure, Measure), "Expecting measure to be a Measure instance.")
        self._integrand = integrand
        self._measure   = measure
        self._repr = "Integral(%r, %r)" % (self._integrand, self._measure)
    
    def reconstruct(self, integrand):
        """Construct a new Integral object with some properties replaced with new values.
        
        Example:
            <a = Integral instance>
            b = a.reconstruct(expand_compounds(a.integrand()))
        """
        return Integral(integrand, self._measure)
    
    def integrand(self):
        "Return the integrand expression, which is an Expr instance."
        return self._integrand
    
    def measure(self):
        "Return the measure associated with this integral."
        return self._measure
    
    def __neg__(self):
        return self.reconstruct(-self._integrand)
    
    def __mul__(self, scalar):
        ufl_assert(is_python_scalar(scalar), "Cannot multiply an integral with non-constant values.")
        return self.reconstruct(scalar*self._integrand)
    
    def __str__(self):
        return "{ %s } * %s" % (self._integrand, self._measure)
    
    def __repr__(self):
        return self._repr
    
    def __eq__(self, other):
        return repr(self) == repr(other)
    
    def __hash__(self):
        return hash(repr)

