#!/usr/bin/env python

"""
The Integral class.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-14-03 -- 2008-16-03"


from output import *
from base import is_true_scalar

from form import Form


class Integral(object):
    """Description of an integral over a single domain."""
    def __init__(self, domain_type, domain_id, integrand = None):
        self.domain_type = domain_type
        self.domain_id   = domain_id
        self.integrand   = integrand
    
    def __mul__(self, other):
        raise RuntimeError("Can't multiply Integral from the left.")
    
    def __rmul__(self, other):
        ufl_assert(self.integrand is None, "Seems to be a bug in Integral.")
        ufl_assert(is_true_scalar(other), "Trying to integrate expression of rank %d with free indices %s." % (other.rank(), repr(other.free_indices())))
        return Form( [Integral(self.domain_type, self.domain_id, other)] )
    
    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        return item in self.integrand
    
    def __str__(self):
        return "Integral(%s, %d, %s)" % (repr(self.domain_type), self.domain_id, self.integrand)
    
    def __repr__(self):
        return "Integral(%s, %s, %s)" % (repr(self.domain_type), repr(self.domain_id), repr(self.integrand))

