"The Form class."

# Copyright (C) 2008-2011 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009-2011.
#
# First added:  2008-03-14
# Last changed: 2011-12-06

import hashlib
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.integral import Integral, Measure, is_scalar_constant_expression
from ufl.equation import Equation


# --- The Form class, representing a complete variational form or functional ---

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = ("_integrals",
                 "_hash", "_signature",
                 "_form_data", "_is_preprocessed",
                 "exterior_facet_domains")

    def __init__(self, integrals):
        self._integrals = tuple(integrals)
        ufl_assert(all(isinstance(itg, Integral) for itg in integrals),
                   "Expecting list of integrals.")
        self._signature = None
        self._hash = None
        self._form_data = None
        self._is_preprocessed = False

    # TODO: Remove these completely!
    def _trigger_domain_error(self):
        msg = "Deprecated: ufl.Form has no properties '*_domains'.\n"
        msg += "To associate domains with a form, use dss = ds[mydomains]; a = f*dss(1)."
        error(msg)
    def _get_domains(self):
        self._trigger_domain_error()
    def _set_domains(self, domains):
        self._trigger_domain_error()
    cell_domains = property(_get_domains, _set_domains)
    exterior_facet_domains = property(_get_domains, _set_domains)
    interior_facet_domains = property(_get_domains, _set_domains)

    def cell(self):
        c = None
        for itg in self._integrals:
            d = itg.integrand().cell()
            if d is not None:
                c = d # Best we found so far
                if not d.is_undefined():
                    # Use the first fully defined cell we find
                    break
        return c

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
        return self.integrals(Measure.CELL)

    def exterior_facet_integrals(self):
        return self.integrals(Measure.EXTERIOR_FACET)

    def interior_facet_integrals(self):
        return self.integrals(Measure.INTERIOR_FACET)

    def macro_cell_integrals(self):
        return self.integrals(Measure.MACRO_CELL)

    def surface_integrals(self):
        return self.integrals(Measure.SURFACE)

    def form_data(self):
        "Return form metadata (None if form has not been preprocessed)"
        return self._form_data

    def compute_form_data(self,
                          object_names=None,
                          common_cell=None,
                          element_mapping=None):
        "Compute and return form metadata"
        if self._form_data is None:
            from ufl.algorithms.preprocess import preprocess
            self._form_data = preprocess(self,
                                         object_names=object_names,
                                         common_cell=common_cell,
                                         element_mapping=element_mapping)
        else:
            self._form_data.validate(object_names=object_names,
                                     common_cell=common_cell,
                                     element_mapping=element_mapping)
        return self.form_data()

    def is_preprocessed(self):
        "Check whether form is preprocessed"
        return self._is_preprocessed

    def __eq__(self, other):
        return Equation(self, other)

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
        "Subtract other form from this one."
        return self + (-other)

    def __neg__(self):
        """Negate all integrals in form.

        This enables the handy "-form" syntax for e.g. the
        linearized system (J, -F) from a nonlinear form F."""
        return Form([-itg for itg in self._integrals])

    def __rmul__(self, scalar):
        "Multiply all integrals in form with constant scalar value."
        # This enables the handy "0*form" or "dt*form" syntax
        ufl_assert(is_scalar_constant_expression(scalar),
                   "A form can only be multiplied by a globally constant scalar expression.")
        return Form([scalar*itg for itg in self._integrals])

    def __mul__(self, coefficient):
        "UFL form operator: Take the action of this form on the given coefficient."
        from ufl.formoperators import action
        return action(self, coefficient)

    def __str__(self):
        if self._integrals:
            return "\n  +  ".join(str(itg) for itg in self._integrals)
        else:
            return "<empty Form>"

    def _compute_signature(self, reprstring=None):
        if self._signature is None:
            if 0:
                if reprstring is None:
                    reprstring = repr(self)
                self._signature = hashlib.sha512(reprstring).hexdigest()
            else:
                from ufl.algorithms.signature import compute_form_signature
                self._signature = compute_form_signature(self)

    def __repr__(self):
        r = "Form([%s])" % ", ".join(repr(itg) for itg in self._integrals)
        # Compute signature now that we have the expensive repr string available anyway
        self._compute_signature(r)
        return r

    def __hash__(self):
        if self._hash is None:
            hashdata = tuple((hash(itg.integrand()), hash(itg.measure())) for itg in self._integrals)
            self._hash = hash(hashdata)
        return self._hash

    def signature(self):
        self._compute_signature()
        assert self._signature
        return self._signature

