"""The Integral class."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-09-26"

# Modified by Anders Logg, 2008

from .output import ufl_assert, ufl_error
from .base import is_true_scalar
from .form import Form

class Integral(object):
    """Description of an integral over a single domain."""
    __slots__ = ("_domain_type", "_domain_id", "_integrand")
    def __init__(self, domain_type, domain_id, integrand = None):
        self._domain_type = domain_type
        self._domain_id   = domain_id
        self._integrand   = integrand
    
    def domain_type(self):
        'Return the domain type ("cell", "exterior_facet" or "interior_facet").'
        return self._domain_type
    
    def domain_id(self):
        "Return the domain id (integer)."
        return self._domain_id
    
    def integrand(self):
        "Return the integrand expression, a UFLObject."
        return self._integrand
    
    def __mul__(self, other):
        ufl_error("Can't multiply Integral from the right.")
    
    def __rmul__(self, other):
        ufl_assert(self._integrand is None, "Seems to be a bug in Integral.")
        ufl_assert(is_true_scalar(other),   
            "Trying to integrate expression of rank %d with free indices %r." \
            % (other.rank(), other.free_indices()))
        return Form( [Integral(self._domain_type, self._domain_id, other)] )
    
    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        return item in self._integrand

    def __call__(self, domain_id):
        "Return integral of same type on given sub domain"
        return Integral(self._domain_type, domain_id)
    
    def __str__(self):
        d = { "cell": "dx",
              "exterior_facet": "ds",
              "interior_facet": "dS"
            }[self._domain_type]
        return "{ %s } * %s%d" % (self._integrand, d, self._domain_id,)
    
    def __repr__(self):
        return "Integral(%r, %r, %r)" % (self._domain_type, self._domain_id, self._integrand)
    
    def __eq__(self, other):
        return repr(self) == repr(other)
    
    def __hash__(self):
        return hash((self._domain_type, self._domain_id, id(self._integrand)))
    
