"""Functions to check the validity of forms."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Modified by Anders Logg, 2008-2009.
# Modified by Mehdi Nikbakht, 2010.
#
# First added:  2008-03-14
# Last changed: 2012-04-12

from ufl.log import warning, error

# UFL classes
from ufl.form import Form
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.constantvalue import is_true_ufl_scalar
from ufl.integral import Measure

# UFL algorithms
from ufl.algorithms.traversal import iter_expressions, traverse_terminals
from ufl.algorithms.propagate_restrictions import check_restrictions

def validate_form(form): # TODO: Can we make this return a list of errors instead of raising exception?
    """Performs all implemented validations on a form. Raises exception if something fails."""
    errors = []
    warnings = []

    if not isinstance(form, Form):
        msg = "Validation failed, not a Form:\n%s" % repr(form)
        error(msg)
        #errors.append(msg)
        #return errors

    # FIXME: Add back check for multilinearity
    # Check that form is multilinear
    #if not is_multilinear(form):
    #    errors.append("Form is not multilinear in arguments.")

    # FIXME DOMAIN: Add check for consistency between domains somehow
    domains = set(t.domain()
                  for e in iter_expressions(form)
                  for t in traverse_terminals(e)) - set((None,))
    if not domains:
        errors.append("Missing domain definition in form.")

    top_domains = set(dom.top_domain() for dom in domains if dom is not None)
    if not top_domains:
        errors.append("Missing domain definition in form.")
    elif len(top_domains) > 1:
        warnings.append("Multiple top domain definitions in form: %s" % str(top_domains))

    # Check that cell is the same everywhere
    cells = set(dom.cell() for dom in top_domains) - set((None,))
    if not cells:
        errors.append("Missing cell definition in form.")
    elif len(cells) > 1:
        errors.append("Multiple cell definitions in form: %s" % str(cells))

    # Check that no Coefficient or Argument instance
    # have the same count unless they are the same
    coefficients = {}
    arguments = {}
    for e in iter_expressions(form):
        for f in traverse_terminals(e):
            if isinstance(f, Coefficient):
                c = f.count()
                if c in coefficients:
                    g = coefficients[c]
                    if not f is g:
                        errors.append("Found different Coefficients with " + \
                                   "same count: %s and %s." % (repr(f), repr(g)))
                else:
                    coefficients[c] = f

            elif isinstance(f, Argument):
                c = f.count()
                if c in arguments:
                    g = arguments[c]
                    if not f is g:
                        if c == -2: msg = "TestFunctions"
                        elif c == -1: msg = "TrialFunctions"
                        else: msg = "Arguments with same count"
                        msg = "Found different %s: %s and %s." % (msg, repr(f), repr(g))
                        errors.append(msg)
                else:
                    arguments[c] = f

    # Check that all integrands are scalar
    for expression in iter_expressions(form):
        if not is_true_ufl_scalar(expression):
            errors.append("Found non-scalar integrand expression:\n%s\n%s" % \
                              (str(expression), repr(expression)))

    # Check that restrictions are permissible
    for integral in form.integrals():
        # Only allow restricitions on interior facet integrals and surface measures
        if integral.measure().domain_type() in (Measure.INTERIOR_FACET, Measure.SURFACE):
            check_restrictions(integral.integrand(), True)
        else:
            check_restrictions(integral.integrand(), False)

    # Raise exception with all error messages
    # TODO: Return errors list instead, need to collect messages from all validations above first.
    if errors:
        final_msg = 'Found errors in validation of form:\n%s' % '\n\n'.join(errors)
        error(final_msg)
