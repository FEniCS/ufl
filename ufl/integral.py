"""The Integral class."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-20"


from output import *
from base import is_true_scalar
from form import Form


class Integral(object):
    """Description of an integral over a single domain."""
    def __init__(self, domain_type, domain_id, integrand = None):
        self._domain_type = domain_type
        self._domain_id   = domain_id
        self._integrand   = integrand
    
    def __mul__(self, other):
        raise RuntimeError("Can't multiply Integral from the left.")
    
    def __rmul__(self, other):
        ufl_assert(self._integrand is None, "Seems to be a bug in Integral.")
        ufl_assert(is_true_scalar(other), "Trying to integrate expression of rank %d with free indices %s." % (other.rank(), repr(other.free_indices())))
        return Form( [Integral(self._domain_type, self._domain_id, other)] )
    
    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        return item in self._integrand
    
    def __str__(self):
        d = { "cell": "dx",
              "exterior_facet": "ds",
              "interior_facet": "dS"
            }[self._domain_type]
        return "{ %s } * %s%d" % (str(self._integrand), d, self._domain_id,)
    
    def __repr__(self):
        return "Integral(%s, %s, %s)" % (repr(self._domain_type), repr(self._domain_id), repr(self._integrand))

