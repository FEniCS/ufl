"""The Form class."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__    = "2008-03-14 -- 2008-09-17"

from .output import ufl_assert

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = ("_integrals", "_repr", "_hash", "_str", "_metadata")
    def __init__(self, integrals):
        self._integrals = integrals
        self._str = None
        self._repr = None
        self._hash = None
        self._metadata = ""
    
    def _get_integrals(self, domain_type):
        return tuple([itg for itg in self._integrals if itg._domain_type == domain_type])
    
    def cell_integrals(self):
        return self._get_integrals("cell")
    
    def exterior_facet_integrals(self):
        return self._get_integrals("exterior_facet")
    
    def interior_facet_integrals(self):
        return self._get_integrals("interior_facet")
    
    def _add(self, other, sign):
        # Start with integrals in self
        newintegrals = list(self._integrals)
        
        # Build domain to index map
        dom2idx = {}
        for itg in newintegrals:
            dom = (itg._domain_type, itg._domain_id)
            ufl_assert(dom not in dom2idx, "Form invariant breached.")
            dom2idx[dom] = len(dom2idx)
        
        # Append other integrals to list or add integrands to existing integrals
        for itg in other._integrals:
            dom = (itg._domain_type, itg._domain_id)
            if dom in dom2idx:
                idx = dom2idx[dom]
                prev_itg = newintegrals[idx]
                integrand = prev_itg._integrand
                if sign == -1:
                    integrand += -1*itg._integrand
                else:
                    integrand += itg._integrand
                c = prev_itg.__class__
                sum_itg = c(prev_itg._domain_type, prev_itg._domain_id, integrand)
                newintegrals[idx] = sum_itg
            else:
                dom2idx[dom] = len(newintegrals)
                newintegrals.append(itg)
        
        return Form(newintegrals)
    
    def __add__(self, other):
        return self._add(other, +1)
    
    def __sub__(self, other):
        return self._add(other, -1)
    
    def __str__(self):
        if self._str is None:
            self._str = "  +  ".join(str(itg) for itg in self._integrals) 
        return self._str
    
    def __repr__(self):
        if self._repr is None:
            self._repr = "Form([%s])" % ", ".join(repr(itg) for itg in self._integrals)
        return self._repr

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Form):
            return False
        return repr(self) == repr(other)
    
    def signature(self):
        return "%s%s" % (repr(self), self._metadata)

