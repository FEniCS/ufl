# -*- coding: utf-8 -*-
"The Form class."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

import hashlib
from itertools import chain
from collections import defaultdict
from ufl.log import error, deprecate
from ufl.assertions import ufl_assert
import ufl.measure
from ufl.integral import Integral, Measure
from ufl.checks import is_scalar_constant_expression
from ufl.equation import Equation
from ufl.core.expr import Expr
from ufl.constantvalue import Zero
from ufl.protocols import id_or_none
from ufl.coefficient import Coefficient

# Export list for ufl.classes
__all_classes__ = ["Form"]

# --- The Form class, representing a complete variational form or functional ---

def _sorted_integrals(integrals):
    """Sort integrals by domain id, integral type, subdomain id
    for a more stable signature computation."""

    # Group integrals in multilevel dict by keys [domain][integral_type][subdomain_id]
    integrals_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for integral in integrals:
        d = integral.ufl_domain()
        if d is None:
            error("Each integral in a form must have a uniquely defined integration domain.")
        it = integral.integral_type()
        si = integral.subdomain_id()
        integrals_dict[d][it][si] += [integral]

    all_integrals = []

    # Order integrals canonically to increase signature stability
    for d in sorted(integrals_dict): # Assuming Domain is sortable
        for it in sorted(integrals_dict[d]): # str is sortable
            for si in sorted(integrals_dict[d][it], key=lambda x: (type(x).__name__, x)): # int/str are sortable
                unsorted_integrals = integrals_dict[d][it][si]
                # TODO: At this point we could order integrals by metadata and integrand,
                #       or even add the integrands with the same metadata. This is done
                #       in accumulate_integrands_with_same_metadata in algorithms/domain_analysis.py
                #       and would further increase the signature stability.
                all_integrals.extend(unsorted_integrals)
                #integrals_dict[d][it][si] = unsorted_integrals

    return tuple(all_integrals) #, integrals_dict

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = (
        # --- List of Integral objects (a Form is a sum of these Integrals, everything else is derived)
        "_integrals",
        # --- Internal variables for caching various data
        "_integration_domains",
        "_domain_numbering",
        "_subdomain_data",
        "_arguments",
        "_coefficients",
        "_coefficient_numbering",
        "_hash",
        "_signature",
        # --- Dict that external frameworks can place framework-specific
        #     data in to be carried with the form
        #     Never use this internally in ufl!
        "_cache",
        )

    def __init__(self, integrals):
        # Basic input checking (further compatibilty analysis happens later)
        ufl_assert(all(isinstance(itg, Integral) for itg in integrals),
                   "Expecting list of integrals.")

        # Store integrals sorted canonically to increase signature stability
        self._integrals = _sorted_integrals(integrals)

        # Internal variables for caching domain data
        self._integration_domains = None
        self._domain_numbering = None

        # Internal variables for caching subdomain data
        self._subdomain_data = None

        # Internal variables for caching form argument data
        self._arguments = None
        self._coefficients = None
        self._coefficient_numbering = None

        # Internal variables for caching of hash and signature after first request
        self._hash = None
        self._signature = None

        # Never use this internally in ufl!
        self._cache = {}

    # --- Accessor interface ---

    def integrals(self):
        "Return a sequence of all integrals in form."
        return self._integrals

    def integrals_by_type(self, integral_type):
        "Return a sequence of all integrals with a particular domain type."
        return tuple(integral for integral in self.integrals()
                     if integral.integral_type() == integral_type)

    #def integrals_dict(self):
    #    "Returns a mapping on the form { domain: { integral_type: { subdomain_id: integral_list } } }."
    #    return self._integrals_dict

    def empty(self):
        "Returns whether the form has no integrals."
        return self.integrals() == ()

    def domains(self):
        deprecate("Form.domains() is deprecated, please use .ufl_domains() instead.")
        return self.ufl_domains()

    def ufl_domains(self):
        """Return the geometric integration domains occuring in the form.

        NB! This does not include domains of coefficients defined on other meshes.

        The return type is a tuple even if only a single domain exists.
        """
        if self._integration_domains is None:
            self._analyze_domains()
        return self._integration_domains

    def cell(self):
        deprecate("Form.cell() is deprecated, please use .ufl_cell() instead.")
        return self.ufl_cell()

    def domain(self):
        deprecate("Form.domain() is deprecated, please use .ufl_domain() instead.")
        return self.ufl_domain()

    def ufl_cell(self):
        "Return the single cell this form is defined on, fails if multiple cells are found."
        return self.ufl_domain().ufl_cell()

    def ufl_domain(self):
        """Return the single geometric integration domain occuring in the form.

        Fails if multiple domains are found.

        NB! This does not include domains of coefficients defined on other
        meshes, look at form data for that additional information.
        """
        # Collect all domains
        domains = self.ufl_domains()
        # Check that all are equal TODO: don't return more than one if all are equal?
        ufl_assert(all(domain == domains[0] for domain in domains),
                   "Calling Form.ufl_domain() is only valid if all integrals share domain.")
        # Return the one and only domain
        return domains[0]

    def geometric_dimension(self):
        "Return the geometric dimension shared by all domains and functions in this form."
        gdims = tuple(set(domain.geometric_dimension() for domain in self.ufl_domains()))
        ufl_assert(len(gdims) == 1,
                  "Expecting all domains and functions in a form to share geometric dimension, got %s." % str(tuple(sorted(gdims))))
        return gdims[0]

    def domain_numbering(self):
        "Return a contiguous numbering of domains in a mapping { domain: number }."
        if self._domain_numbering is None:
            self._analyze_domains()
        return self._domain_numbering

    def subdomain_data(self):
        "Returns a mapping on the form { domain: { integral_type: subdomain_data } }."
        if self._subdomain_data is None:
            self._analyze_subdomain_data()
        return self._subdomain_data

    def max_subdomain_ids(self):
        "Returns a mapping on the form { domain: { integral_type: max_subdomain_id } }."
        if self._max_subdomain_ids is None:
            self._analyze_subdomain_data()
        return self._max_subdomain_ids

    def arguments(self):
        "Return all Argument objects found in form."
        if self._arguments is None:
            self._analyze_form_arguments()
        return self._arguments

    def coefficients(self):
        "Return all Coefficient objects found in form."
        if self._coefficients is None:
            self._analyze_form_arguments()
        return self._coefficients

    def coefficient_numbering(self):
        "Return a contiguous numbering of coefficients in a mapping { coefficient: number }."
        if self._coefficient_numbering is None:
            self._analyze_form_arguments()
        return self._coefficient_numbering

    def signature(self):
        "Signature for use with jit cache (independent of incidental numbering of indices etc.)"
        if self._signature is None:
            self._compute_signature()
        return self._signature

    # --- Operator implementations ---

    def __hash__(self):
        "Hash code for use in dicts (includes incidental numbering of indices etc.)"
        if self._hash is None:
            self._hash = hash(tuple(hash(itg) for itg in self.integrals()))
        return self._hash

    def __eq__(self, other):
        """Delayed evaluation of the == operator!

        Just 'lhs_form == rhs_form' gives an Equation,
        while 'bool(lhs_form == rhs_form)' delegates
        to lhs_form.equals(rhs_form).
        """
        return Equation(self, other)

    def __ne__(self, other):
        "Immediate evaluation of the != operator (as opposed to the == operator)."
        return not self.equals(other)

    def equals(self, other):
        "Evaluate 'bool(lhs_form == rhs_form)'."
        if type(other) != Form:
            return False
        if len(self._integrals) != len(other._integrals):
            return False
        if hash(self) != hash(other):
            return False
        return all(a == b for a, b in zip(self._integrals, other._integrals))

    def __radd__(self, other):
        # Ordering of form additions make no difference
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Form):
            # Add integrals from both forms
            return Form(list(chain(self.integrals(), other.integrals())))

        elif isinstance(other, (int, float)) and other == 0:
            # Allow adding 0 or 0.0 as a no-op, needed for sum([a,b])
            return self

        elif isinstance(other, Zero) and not (other.ufl_shape or other.ufl_free_indices):
            # Allow adding ufl Zero as a no-op, needed for sum([a,b])
            return self

        else:
            # Let python protocols do their job if we don't handle it
            return NotImplemented

    def __sub__(self, other):
        "Subtract other form from this one."
        return self + (-other)

    def __rsub__(self, other):
        "Subtract this form from other."
        return other + (-self)

    def __neg__(self):
        """Negate all integrals in form.

        This enables the handy "-form" syntax for e.g. the
        linearized system (J, -F) from a nonlinear form F."""
        return Form([-itg for itg in self.integrals()])

    def __rmul__(self, scalar):
        "Multiply all integrals in form with constant scalar value."
        # This enables the handy "0*form" or "dt*form" syntax
        if is_scalar_constant_expression(scalar):
            return Form([scalar*itg for itg in self.integrals()])
        return NotImplemented

    def __mul__(self, coefficient):
        "UFL form operator: Take the action of this form on the given coefficient."
        if isinstance(coefficient, Expr): #Coefficient): # TODO: Check whatever makes sense
            from ufl.formoperators import action
            return action(self, coefficient)
        return NotImplemented

    # --- String conversion functions, for UI purposes only ---

    def __str__(self):
        "Compute shorter string representation of form. This can be huge for complicated forms."
        # TODO: Add warning here to check if anyone actually calls it in libraries
        s = "\n  +  ".join(str(itg) for itg in self.integrals())
        return s or "<empty Form>"

    def __repr__(self):
        "Compute repr string of form. This can be huge for complicated forms."
        # TODO: Add warning here to check if anyone actually calls it in libraries
        # Not caching this because it can be huge
        r = "Form([%s])" % ", ".join(repr(itg) for itg in self.integrals())
        return r

    def x_repr_latex_(self): # TODO: This works, but enable when form latex rendering is fixed
        from ufl.algorithms import ufl2latex
        return "$$%s$$" % ufl2latex(self)

    def x_repr_png_(self): # TODO: This works, but enable when form latex rendering is fixed
        from IPython.lib.latextools import latex_to_png
        return latex_to_png(self._repr_latex_())

    # --- Analysis functions, precomputation and caching of various quantities ---

    def _analyze_domains(self):
        from ufl.domain import join_domains, sort_domains

        # Collect unique integration domains
        integration_domains = join_domains([itg.ufl_domain() for itg in self._integrals])

        # Make canonically ordered list of the domains
        self._integration_domains = sort_domains(integration_domains)

        # TODO: Not including domains from coefficients and arguments here, may need that later
        self._domain_numbering = dict((d, i) for i, d in enumerate(self._integration_domains))

    def _analyze_subdomain_data(self):
        integration_domains = self.ufl_domains()
        integrals = self.integrals()

        # Make clear data structures to collect subdomain data in
        subdomain_data = {}
        for domain in integration_domains:
            subdomain_data[domain] = {}

        for integral in integrals:
            # Get integral properties
            domain = integral.ufl_domain()
            it = integral.integral_type()
            sd = integral.subdomain_data()

            # Collect subdomain data
            data = subdomain_data[domain].get(it)
            if data is None:
                subdomain_data[domain][it] = sd
            elif sd is not None:
                ufl_assert(data.ufl_id() == sd.ufl_id(), "Integrals in form have different subdomain_data objects.")
        self._subdomain_data = subdomain_data

    def _analyze_form_arguments(self):
        "Analyze which Argument and Coefficient objects can be found in the form."
        from ufl.algorithms.analysis import extract_arguments_and_coefficients
        arguments, coefficients = extract_arguments_and_coefficients(self)

        # Define canonical numbering of arguments and coefficients
        self._arguments = tuple(sorted(set(arguments), key=lambda x: x.number()))
        self._coefficients = tuple(sorted(set(coefficients), key=lambda x: x.count()))
        self._coefficient_numbering = dict((c, i) for i, c in enumerate(self._coefficients))

    def _compute_renumbering(self):
        # Include integration domains and coefficients in renumbering
        dn = self.domain_numbering()
        cn = self.coefficient_numbering()
        renumbering = {}
        renumbering.update(dn)
        renumbering.update(cn)

        # Add domains of coefficients, these may include domains not among integration domains
        k = len(dn)
        for c in cn:
            d = c.ufl_domain()
            if d is not None and d not in renumbering:
                renumbering[d] = k
                k += 1

        return renumbering

    def _compute_signature(self):
        from ufl.algorithms.signature import compute_form_signature
        self._signature = compute_form_signature(self, self._compute_renumbering())


def as_form(form):
    "Convert to form if not a form, otherwise return form."
    if not isinstance(form, Form):
        error("Unable to convert object to a UFL form: %s" % repr(form))
    return form


def replace_integral_domains(form, common_domain): # TODO: Move elsewhere
    """Given a form and a domain, assign a common integration domain to all integrals.

    Does not modify the input form (Form should always be immutable).
    This is to support ill formed forms with no domain specified,
    some times occuring in pydolfin, e.g. assemble(1*dx, mesh=mesh).
    """
    domains = form.ufl_domains()
    if common_domain is not None:
        gdim = common_domain.geometric_dimension()
        tdim = common_domain.topological_dimension()
        ufl_assert(all((gdim == domain.geometric_dimension() and
                        tdim == domain.topological_dimension())
                        for domain in domains),
            "Common domain does not share dimensions with form domains.")

    reconstruct = False
    integrals = []
    for itg in form.integrals():
        domain = itg.ufl_domain()
        if domain != common_domain:
            itg = itg.reconstruct(domain=common_domain)
            reconstruct = True
        integrals.append(itg)
    if reconstruct:
        form = Form(integrals)
    return form
