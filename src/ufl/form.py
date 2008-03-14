#!/usr/bin/env python

"""
The Form class.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-14-03"

from output import *
from base import is_true_scalar


class Form:
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    def __init__(self, integrals):    
        self.integrals = integrals
    
    def _integrals(self, domain_type):
        return tuple([i for i in self.integrals if i.domain_type == domain_type])
    
    def cell_integrals(self):
        return self._integrals("cell")
    
    def exterior_facet_integrals(self):
        return self._integrals("exterior_facet")
    
    def interior_facet_integrals(self):
        return self._integrals("interior_facet")

    def __contains__(self, item):
        """Return wether item is in the UFL expression tree. If item is a str, it is assumed to be a repr."""
        return any(item in itg for itg in self.integrals)
    
    def __add__(self, other):
        return Form(self.integrals + other.integrals)
    
    def __str__(self):
        return "Form([%s])" % ", ".join(str(i) for i in self.integrals)
    
    def __repr__(self):
        return "Form([%s])" % ", ".join(repr(i) for i in self.integrals)

