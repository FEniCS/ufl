"The Form class."

__authors__ = "Martin Sandve Alnes"
__date__    = "2008-03-14 -- 2009-02-16"

from ufl.assertions import ufl_assert
from ufl.constantvalue import as_ufl, is_python_scalar
from ufl.sorting import cmp_expr

# --- The Form class, representing a complete variational form or functional ---

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = ("_integrals", "_repr", "_hash", "_str",)
    def __init__(self, integrals):
        self._integrals = tuple(integrals)
        self._str = None
        self._repr = None
        self._hash = None
    
    def cell(self):
        return self._integrals[0].integrand().cell()
    
    def integrals(self):
        return self._integrals
    
    def _get_integrals(self, domain_type):
        return tuple(itg for itg in self._integrals if itg.measure().domain_type() == domain_type)
    
    def cell_integrals(self):
        from ufl.integral import Measure
        return self._get_integrals(Measure.CELL)
    
    def exterior_facet_integrals(self):
        from ufl.integral import Measure
        return self._get_integrals(Measure.EXTERIOR_FACET)
    
    def interior_facet_integrals(self):
        from ufl.integral import Measure
        return self._get_integrals(Measure.INTERIOR_FACET)
    
    def __add__(self, other):
        
        # --- Add integrands of integrals with the same measure
        
        # Start with integrals in self
        newintegrals = list(self._integrals)
        
        # Build mapping: (measure -> self._integrals index)
        measure2idx = {}
        for i, itg in enumerate(newintegrals):
            ufl_assert(itg.measure() not in measure2idx, "Form invariant breached.")
            measure2idx[itg.measure()] = i
        
        for itg in other._integrals:
            idx = measure2idx.get(itg.measure())
            if idx is None:
                # Append integral with new measure to list 
                idx = len(newintegrals)
                measure2idx[itg.measure()] = idx
                newintegrals.append(itg)
            else:
                # Accumulate integrands with same measure
                a = newintegrals[idx].integrand()
                b = itg.integrand()
                # Invariant ordering of terms (shouldn't Sum fix this?)
                #if cmp_expr(a, b) > 0:
                #    a, b = b, a
                newintegrals[idx] = itg.reconstruct(a + b)
        
        return Form(newintegrals)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        # This enables the handy "-form" syntax for e.g. the linearized system (J, -F) from a nonlinear form F
        return Form([-itg for itg in self._integrals])
    
    def __rmul__(self, scalar):
        # This enables the handy "0*form" syntax
        ufl_assert(is_python_scalar(scalar), "Only multiplication by scalar literals currently supported.")
        return Form([scalar*itg for itg in self._integrals])
    
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
            self._hash = hash(tuple(type(itg) for itg in self._integrals))
            #self._hash = hash(repr(self))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Form):
            return False
        return repr(self) == repr(other)
    
    def signature(self):
        return "%s" % (repr(self), )

