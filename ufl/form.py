"The Form class."

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__    = "2008-03-14 -- 2009-12-08"

# Modified by Anders Logg, 2009-2011.

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.constantvalue import as_ufl, is_python_scalar
from ufl.sorting import cmp_expr
from ufl.integral import Integral, Measure

# --- The Form class, representing a complete variational form or functional ---

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = ("_integrals", "_repr", "_hash", "_str", "_form_data", "_is_preprocessed",
                 "cell_domains", "exterior_facet_domains", "interior_facet_domains")

    # Note: cell_domains, exterior_facet_domains and interior_facet_domains
    # are used by DOLFIN to pass data to the assembler. They can otherwise
    # safely be ignored.

    def __init__(self, integrals):
        self._integrals = tuple(integrals)
        ufl_assert(all(isinstance(itg, Integral) for itg in integrals), "Expecting list of integrals.")
        self._str = None
        self._repr = None
        self._hash = None
        self._form_data = None
        self._is_preprocessed = False
        self.cell_domains = None
        self.exterior_facet_domains = None
        self.interior_facet_domains = None

    def cell(self):
        for itg in self._integrals:
            c = itg.integrand().cell()
            if c is not None:
                return c
        return None

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

    def macro_cell_integrals(self):
        from ufl.integral import Measure
        return self.integrals(Measure.MACRO_CELL)

    def surface_integrals(self):
        from ufl.integral import Measure
        return self.integrals(Measure.SURFACE)

    def form_data(self):
        "Return form metadata (None if form has not been preprocessed)"
        return self._form_data

    def compute_form_data(self, object_names={}, common_cell=None, element_mapping={}):
        "Compute and return form metadata"
        if self._form_data is None:
            from ufl.algorithms.preprocess import preprocess
            self._form_data = preprocess(self, object_names, common_cell, element_mapping)
        return self.form_data()

    def is_preprocessed(self):
        "Check whether form is preprocessed"
        return self._is_preprocessed

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

    def __mul__(self, function):
        "The action of this form on the given function."
        from ufl.formoperators import action
        return action(self, function)

    def __str__(self):
        if self._str is None:
            if self._integrals:
                self._str = "\n  +  ".join(str(itg) for itg in self._integrals)
            else:
                self._str = "<empty Form>"
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
