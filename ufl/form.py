"""The Form class."""

__authors__ = "Martin Sandve Alnes"
__date__    = "2008-03-14 -- 2008-08-12"


class Form:
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    def __init__(self, integrals):
        self._integrals = integrals
    
    def _get_integrals(self, domain_type):
        return tuple([itg for itg in self._integrals if itg._domain_type == domain_type])
    
    def cell_integrals(self):
        return self._get_integrals("cell")
    
    def exterior_facet_integrals(self):
        return self._get_integrals("exterior_facet")
    
    def interior_facet_integrals(self):
        return self._get_integrals("interior_facet")
    
    def __add__(self, other):
        oldintegrals = self._integrals + other._integrals
        newintegrals = []
        dom2idx = {}
        k = 0
        for i, itg in enumerate(oldintegrals):
            dom = (itg._domain_type, itg._domain_id)
            if dom in dom2idx:
                idx = dom2idx[dom]
                itg1 = newintegrals[idx]
                newintegrals[idx] = itg.__class__(itg1._domain_type, itg1._domain_id, \
                                             itg1._integrand + itg._integrand)
            else:
                dom2idx[dom] = k
                newintegrals.append(itg)
                k += 1
                assert len(newintegrals) == k
        return Form(newintegrals)
        #return Form(self._integrals + other._integrals)
    
    def __str__(self):
        return "  +  ".join(str(itg) for itg in self._integrals)
    
    def __repr__(self):
        return "Form([%s])" % ", ".join(repr(itg) for itg in self._integrals)
