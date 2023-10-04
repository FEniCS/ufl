"""FormData class easy for collecting of various data about a form."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008.

from ufl.utils.formatting import lstr, tstr, estr


class FormData(object):
    """Class collecting various information extracted from a Form by calling preprocess."""

    def __init__(self):
        """Create empty form data for given form."""

    def __str__(self):
        """Return formatted summary of form data."""
        types = sorted(self.max_subdomain_ids.keys())
        geometry = (("Geometric dimension", self.geometric_dimension), )
        subdomains = tuple((f"Number of {integral_type} subdomains", self.max_subdomain_ids[integral_type])
                           for integral_type in types)
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
