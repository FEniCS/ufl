"""The Integral class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-01-09"

# Modified by Anders Logg, 2008

from ufl.output import ufl_assert, ufl_error
from ufl.scalar import is_true_ufl_scalar

class Integral(object):
    """Description of an integral over a single domain."""
    __slots__ = ("_domain_type", "_domain_id", "_integrand", "_metadata")
    def __init__(self, domain_type, domain_id, integrand = None, metadata = None):
        self._domain_type = domain_type
        self._domain_id   = domain_id
        self._integrand   = integrand
        self._metadata    = metadata
    
    def reconstruct(self, domain_id=None, integrand=None, metadata=None):
        """Construct a new Integral object with some properties replaced with new values.
        
        Example:
            <a = Integral instance>
            b = a.reconstruct(integrand=expand_compounds(a))
            c = a.reconstruct(metadata={"quadrature_order":3})
        """
        if domain_id is None: domain_id = self._domain_id
        if integrand is None: integrand = self._integrand
        if metadata  is None: metadata  = self._metadata
        return Integral(self._domain_type, domain_id, integrand, metadata)
    
    # Enumeration of valid domain types
    CELL = "cell"
    EXTERIOR_FACET = "exterior_facet"
    INTERIOR_FACET = "interior_facet"
    
    def domain_type(self):
        'Return the domain type ("cell", "exterior_facet" or "interior_facet").'
        return self._domain_type
    
    def domain_id(self):
        "Return the domain id (integer)."
        return self._domain_id
    
    def integrand(self):
        "Return the integrand expression, a Expr."
        return self._integrand
    
    def metadata(self):
        "Return the integral metadata. What this can be is currently undefined." # TODO!
        return self._metadata
    
    def __call__(self, domain_id, metadata=None): # TODO: Define how metadata should represent integration data here: quadrature_degree, quadrature_rule, ...
        "Return integral of same type on given sub domain"
        return self.reconstruct(domain_id=domain_id, metadata=metadata)
    
    def __mul__(self, other):
        ufl_error("Can't multiply Integral from the right.")
    
    def __rmul__(self, integrand):
        ufl_assert(self._integrand is None, "Integrand is already defined, can't integrate twice.")
        ufl_assert(is_true_ufl_scalar(integrand),   
            "Trying to integrate expression of rank %d with free indices %r." \
            % (integrand.rank(), integrand.free_indices()))
        from ufl.form import Form
        return Form( [self.reconstruct(integrand=integrand)] )
    
    def __neg__(self):
        return Integral(self._domain_type, self._domain_id, -self._integrand)
    
    def __str__(self):
        d = { Integral.CELL: "dx",
              Integral.EXTERIOR_FACET: "ds",
              Integral.INTERIOR_FACET: "dS"
            }[self._domain_type]
        metastring = "" if self._metadata is None else ("[%s]" % repr(self._metadata))
        return "{ %s } * %s%d%s" % (self._integrand, d, self._domain_id, metastring)
    
    def __repr__(self):
        return "Integral(%r, %r, %r, %r)" % (self._domain_type, self._domain_id, self._integrand, self._metadata)
    
    def __eq__(self, other):
        return repr(self) == repr(other)
    
    def __hash__(self):
        return hash((self._domain_type, self._domain_id, id(self._integrand)))
    
