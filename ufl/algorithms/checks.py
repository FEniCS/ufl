# -*- coding: utf-8 -*-
"""Functions to check the validity of forms."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009.
# Modified by Mehdi Nikbakht, 2010.

from ufl.algorithms.check_restrictions import check_restrictions
# UFL algorithms
from ufl.algorithms.traversal import iter_expressions
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.constantvalue import is_true_ufl_scalar
# UFL classes
from ufl.core.expr import ufl_err_str
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.domain import extract_unique_domain
from ufl.form import Form


def validate_form(form):  # TODO: Can we make this return a list of errors instead of raising exception?
    """Performs all implemented validations on a form. Raises exception if something fails."""
    errors = []

    if not isinstance(form, Form):
        raise ValueError(f"Validation failed, not a Form:\n{ufl_err_str(form)}")

    # FIXME: There's a bunch of other checks we should do here.

    # FIXME: Add back check for multilinearity
    # Check that form is multilinear
    # if not is_multilinear(form):
    #     errors.append("Form is not multilinear in arguments.")

    # FIXME DOMAIN: Add check for consistency between domains somehow
    domains = set(extract_unique_domain(t)
                  for e in iter_expressions(form)
                  for t in traverse_unique_terminals(e)) - {None}
    if not domains:
        errors.append("Missing domain definition in form.")

    # Check that cell is the same everywhere
    cells = set(dom.ufl_cell() for dom in domains) - {None}
    if not cells:
        errors.append("Missing cell definition in form.")
    elif len(cells) > 1:
        errors.append(f"Multiple cell definitions in form: {cells}")

    # Check that no Coefficient or Argument instance have the same
    # count unless they are the same
    coefficients = {}
    arguments = {}
    for e in iter_expressions(form):
        for f in traverse_unique_terminals(e):
            if isinstance(f, Coefficient):
                c = f.count()
                if c in coefficients:
                    g = coefficients[c]
                    if f is not g:
                        errors.append("Found different Coefficients with "
                                      f"same count: {f} and {g}.")
                else:
                    coefficients[c] = f

            elif isinstance(f, Argument):
                n = f.number()
                p = f.part()
                if (n, p) in arguments:
                    g = arguments[(n, p)]
                    if f is not g:
                        if n == 0:
                            msg = "TestFunctions"
                        elif n == 1:
                            msg = "TrialFunctions"
                        else:
                            msg = "Arguments with same number and part"
                        msg = "Found different %s: %s and %s." % (msg, repr(f), repr(g))
                        errors.append(msg)
                else:
                    arguments[(n, p)] = f

    # Check that all integrands are scalar
    for expression in iter_expressions(form):
        if not is_true_ufl_scalar(expression):
            errors.append("Found non-scalar integrand expression: %s\n" %
                          ufl_err_str(expression))

    # Check that restrictions are permissible
    for integral in form.integrals():
        # Only allow restrictions on interior facet integrals and
        # surface measures
        if integral.integral_type().startswith("interior_facet"):
            check_restrictions(integral.integrand(), True)
        else:
            check_restrictions(integral.integrand(), False)

    # Raise exception with all error messages
    # TODO: Return errors list instead, need to collect messages from
    # all validations above first.
    if errors:
        raise ValueError("Found errors in validation of form:\n" + '\n\n'.join(errors))
