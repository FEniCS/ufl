# -*- coding: utf-8 -*-
"""FormData class easy for collecting of various data about a form."""

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
# Modified by Anders Logg, 2008.

from ufl.utils.formatting import lstr, tstr, estr


class FormData(object):
    """
    Class collecting various information extracted from a Form by
    calling preprocess.
    """

    def __init__(self):
        "Create empty form data for given form."

    def __unicode__(self):
        # Only in python 2
        return str(self).decode("utf-8")

    def __bytes__(self):
        # Only in python 3
        return str(self).encode("utf-8")

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

    def __unicode__(self):
        # Only in python 2
        return str(self).decode("utf-8")

    def __bytes__(self):
        # Only in python 3
        return str(self).encode("utf-8")

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
