"The Form class."

__authors__ = "Martin Sandve Alnes"
__date__    = "2008-03-14 -- 2009-03-29"

# Modified by Anders Logg, 2009.

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.constantvalue import as_ufl, is_python_scalar
from ufl.sorting import cmp_expr
from ufl.integral import Integral, Measure
from ufl.operators import inner

# --- The Form class, representing a complete variational form or functional ---

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = ("_integrals", "_repr", "_hash", "_str", "_form_data")

    def __init__(self, integrals):
        #self._integrals = tuple(integrals)
        self._integrals = _extract_integrals(integrals)
        self._str = None
        self._repr = None
        self._hash = None
        self._form_data = None

    def form_data(self):
        if self._form_data is None:
            from ufl.algorithms.formdata import FormData
            self._form_data = FormData(self)
        return self._form_data
    
    def cell(self):
        return self._integrals[0].integrand().cell()
    
    def integral_groups(self):
        """Return a dict, which is a mapping from domains to integrals.
        
        In particular, each key of the dict is a distinct tuple
        (domain_type, domain_id), and each value is a list of
        Integral instances. The Integrals in each list share the
        same domain (the key), but have different measures."""
        d = {}
        for itg in self.integrals():
            m = itg.measure()
            k = (m.domain_type(), m.domain_id())
            l = d.get(k)
            if not l:
                l = []
                d[k] = l
            l.append(itg)
        return d
    
    def integrals(self, domain_type = None):
        if domain_type is None:
            return self._integrals
        return tuple(itg for itg in self._integrals if itg.measure().domain_type() == domain_type)
    
    def measures(self, domain_type = None):
        return tuple(itg.measure() for itg in self.integrals(domain_type))
    
    def domains(self, domain_type = None):
        return tuple((m.domain_type(), m.domain_id()) for m in self.measures(domain_type))
    
    def cell_integrals(self):
        from ufl.integral import Measure
        return self.integrals(Measure.CELL)
    
    def exterior_facet_integrals(self):
        from ufl.integral import Measure
        return self.integrals(Measure.EXTERIOR_FACET)
    
    def interior_facet_integrals(self):
        from ufl.integral import Measure
        return self.integrals(Measure.INTERIOR_FACET)
    
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
            self._str = "\n  +  ".join(str(itg) for itg in self._integrals) 
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

def _extract_integrals(objects):
    "Extract integrals from single integral of tuple of integrals."

    # Make sure we get a list or tuple
    if not isinstance(objects, (list, tuple)):
        objects = [objects]

    # Operands and default measure
    v = w = None
    dx = Measure(Measure.CELL, 0)

    # Iterate over objects and extract integrals
    integrals = []
    for object in objects:

        # Found plain integral, just append
        if isinstance(object, Integral):
            integrals.append(object)

        # Found measure, append inner(v, w)*dm
        elif isinstance(object, Measure):
            dm = object
            if v is None or w is None:
                error("Found measure without matching integrands: " + str(dm))
            else:
                form = inner(v, w)*dm
                integrals += form.integrals()
                v = w = None

        # Found first operand, store v
        elif v is None and w is None:
            v = object

        # Found second operand, store w
        elif w is None:
            w = object

        # Found new operand, assume measure is dx
        elif not v is None and not w is None:
            form = inner(v, w)*dx
            integrals += form.integrals()
            v = object
            w = None

        # Default case, should not get here
        else:
            error("Unable to extract form, expression does not make sense.")

    # Add last inner product if any
    if not v is None and not w is None:
        form = inner(v, w)*dx
        integrals += form.integrals()

    return tuple(integrals)
