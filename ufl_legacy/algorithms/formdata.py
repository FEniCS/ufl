# -*- coding: utf-8 -*-
"""FormData class easy for collecting of various data about a form."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008.

from ufl_legacy.utils.formatting import lstr, tstr, estr


class FormData(object):
    """Class collecting various information extracted from a Form by
    calling preprocess.

    """

    def __init__(self):
        "Create empty form data for given form."

    def __str__(self):
        "Return formatted summary of form data"
        types = sorted(self.max_subdomain_ids.keys())
        geometry = (
            ("Geometric dimension", self.geometric_dimension),
        )
        subdomains = tuple(("Number of %s subdomains" % integral_type,
                            self.max_subdomain_ids[integral_type]) for integral_type in types)
        functions = (
            # Arguments
            ("Rank", self.rank),
            ("Arguments", lstr(self.original_form.arguments())),
            # Coefficients
            ("Number of coefficients", self.num_coefficients),
            ("Coefficients", lstr(self.reduced_coefficients)),
            # Elements
            ("Unique elements", estr(self.unique_elements)),
            ("Unique sub elements", estr(self.unique_sub_elements)),
        )
        return tstr(geometry + subdomains + functions)


class ExprData(object):
    """
    Class collecting various information extracted from a Expr by
    calling preprocess.
    """

    def __init__(self):
        "Create empty expr data for given expr."

    def __str__(self):
        "Return formatted summary of expr data"
        return tstr((("Name", self.name),
                     ("Cell", self.cell),
                     ("Topological dimension", self.topological_dimension),
                     ("Geometric dimension", self.geometric_dimension),
                     ("Rank", self.rank),
                     ("Number of coefficients", self.num_coefficients),
                     ("Arguments", lstr(self.arguments)),
                     ("Coefficients", lstr(self.coefficients)),
                     ("Argument names", lstr(self.argument_names)),
                     ("Coefficient names", lstr(self.coefficient_names)),
                     ("Unique elements", estr(self.unique_elements)),
                     ("Unique sub elements", estr(self.unique_sub_elements)),
                     # FIXME DOMAINS what is "the domain(s)" for an expression?
                     ("Domains", self.domains), ))
