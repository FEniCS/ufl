"The Form class."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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
from ufl.log import error
from ufl.assertions import ufl_assert
import ufl.measure
from ufl.integral import Integral, Measure
from ufl.checks import is_scalar_constant_expression
from ufl.equation import Equation
from ufl.expr import Expr
from ufl.constantvalue import Zero
from ufl.geometry import join_domains
from ufl.protocols import id_or_none

# --- The Form class, representing a complete variational form or functional ---

def integral_sort_key(integral):
    domain = integral.domain()
    label = None if domain is None else domain.label()
    return (str(label), str(integral.integral_type()), str(integral.subdomain_id()))

def replace_integral_domains(form, common_domain): # TODO: Move elsewhere
    """Given a form and a domain, assign a common integration domain to all integrals.

    Does not modify the input form (Form should always be immutable).
    This is to support ill formed forms with no domain specified,
    some times occuring in pydolfin, e.g. assemble(1*dx, mesh=mesh).
    """
    domains = form.domains()
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
        domain = itg.domain()
        if domain is None or domain.label() != common_domain.label():
            itg = itg.reconstruct(domain=common_domain)
            reconstruct = True
        integrals.append(itg)
    if reconstruct:
        form = Form(integrals)
    return form

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = (
        # List of Integral objects (a Form is a sum of these Integrals)
        "_integrals",
        # List of Domain objects (the domains that Integrals of this form integrate over, not including other domains that integrands may reference)
        "_domains",
        # Hash code for use in dicts,
        # including incidental numbering of indices etc.
        "_hash",
        # Signature for use with jit cache,
        # independent of incidental numbering of indices etc.
        "_signature",
        # Cache of preprocess result applied to this form
        "_form_data",
        # Set to true if this form is the result of a preprocess of another form
        "_is_preprocessed",
        )

    def __init__(self, integrals):
        ufl_assert(all(isinstance(itg, Integral) for itg in integrals),
                   "Expecting list of integrals.")

        # Store integral list in canonical ordering
        self._integrals = sorted(integrals, key=integral_sort_key)

        # Collect integration domains and make canonical list of them
        self._domains = join_domains([itg.domain() for itg in integrals])

        # Internal variables for caching
        self._signature = None
        self._hash = None

        # Internal variables for caching preprocessing data
        self._form_data = None
        self._is_preprocessed = False

    def cell(self):
        deprecate("Form.cell() is not well defined and will be removed.")
        domain = self.domain()
        return None if domain is None else domain.cell()

    def domain(self):
        """Return the geometric integration domain occuring in the form.

        NB! This does not include domains of coefficients defined on other
        meshes, look at form data for that additional information.
        """
        deprecate("Form.domain() is not well defined and will be removed.")

        domains = self.domains()

        ufl_assert(all(domain == domains[0] for domain in domains),
                   "Calling Form.domain() is only valid if all integrals share domain.")

        # Need to support missing domain to allow
        # assemble(Constant(1)*dx, mesh=mesh) in dolfin
        return domains[0] if domains else None

    def domains(self):
        """Return the geometric integration domains occuring in the form.

        NB! This does not include domains of coefficients defined on other
        meshes, look at form data for that additional information.

        The return type is a tuple even if only a single domain exists.
        """
        return self._domains

    def integrals(self):
        "Return a sequence of all integrals in form."
        return self._integrals

    def integrals_by_type(self, integral_type):
        "Return a sequence of all integrals with a particular domain type."
        return [integral for integral in self.integrals()
                if integral.integral_type() == integral_type]

    def empty(self):
        return self.integrals() == ()

    def is_preprocessed(self):
        "Return true if this form is the result of a preprocessing of another form."
        return self._is_preprocessed

    def form_data(self):
        "Return form metadata (None if form has not been preprocessed)"
        return self._form_data

    def compute_form_data(self, object_names=None):
        "Compute and return form metadata"

        # TODO: We should get rid of the form data caching, but need to
        #       figure out how to do that and keep pydolfin working properly

        # Only compute form data once, and never on an already processed form
        ufl_assert(not self.is_preprocessed(), "You can not preprocess forms twice.")

        if self._form_data is None:
            from ufl.algorithms.preprocess import preprocess
            self._form_data = preprocess(self, object_names=object_names)

        # Always validate arguments, keeping sure that the validation works
        self._form_data.validate(object_names=object_names)
        return self.form_data()

    def __eq__(self, other):
        """Delayed evaluation of the __eq__ operator!

        Just 'lhs_form == rhs_form' gives an Equation,
        while 'bool(lhs_form == rhs_form)' delegates
        to lhs_form.equals(rhs_form).
        """
        return Equation(self, other)

    def equals(self, other):
        "Evaluate 'bool(lhs_form == rhs_form)'."
        if type(other) != Form:
            return False
        if len(self._integrals) != len(other._integrals):
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

        elif isinstance(other, Zero) and not (other.shape() or other.free_indices()):
            # Allow adding ufl Zero as a no-op, needed for sum([a,b])
            return self

        else:
            # Let python protocols do their job if we don't handle it
            return NotImplemented

    def __sub__(self, other):
        "Subtract other form from this one."
        return self + (-other)

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

    def __str__(self):
        # TODO: Add warning here to check if anyone actually calls it in libraries
        s = "\n  +  ".join(str(itg) for itg in self.integrals())
        return s or "<empty Form>"

    def __repr__(self):
        # TODO: Add warning here to check if anyone actually calls it in libraries
        # Not caching this because it can be huge
        r = "Form([%s])" % ", ".join(repr(itg) for itg in self.integrals())
        return r

    def __hash__(self):
        if self._hash is None:
            hashdata = tuple(hash(itg) for itg in self.integrals())
            self._hash = hash(hashdata)
        return self._hash

    def x_repr_latex_(self): # TODO: This works, but enable when form latex rendering is fixed
        from ufl.algorithms import ufl2latex
        return "$$%s$$" % ufl2latex(self)

    def x_repr_png_(self): # TODO: This works, but enable when form latex rendering is fixed
        from IPython.lib.latextools import latex_to_png
        return latex_to_png(self._repr_latex_())

def as_form(form):
    "Convert to form if not a form, otherwise return form."
    if not isinstance(form, Form):
        error("Unable to convert object to a UFL form: %s" % repr(form))

    return form
