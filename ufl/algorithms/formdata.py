"""FormData class easy for collecting of various data about a form."""

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
# Modified by Anders Logg, 2008.
#
# First added:  2008-09-13
# Last changed: 2010-02-07

from ufl.common import lstr, tstr, estr

class FormData(object):
    """
    Class collecting various information extracted from a Form by
    calling preprocess.
    """

    def __init__(self):
        "Create empty form data for given form."

    def __str__(self):
        "Return formatted summary of form data"
        return tstr((("Name",                               self.name),
                     ("Rank",                               self.rank),
                     ("Cell",                               self.cell),
                     ("Topological dimension",              self.topological_dimension),
                     ("Geometric dimension",                self.geometric_dimension),
                     ("Number of facets",                   self.num_facets),
                     ("Number of coefficients",             self.num_coefficients),
                     ("Number of cell domains",             self.num_cell_domains),
                     ("Number of exterior facet domains",   self.num_exterior_facet_domains),
                     ("Number or interior facet domains",   self.num_interior_facet_domains),
                     ("Number of macro cell domains",       self.num_macro_cell_domains),
                     ("Number of surface domains",          self.num_surface_domains),
                     ("Arguments",                          lstr(self.arguments)),
                     ("Coefficients",                       lstr(self.coefficients)),
                     ("Argument names",                     lstr(self.argument_names)),
                     ("Coefficient names",                  lstr(self.coefficient_names)),
                     ("Unique elements",                    estr(self.unique_elements)),
                     ("Unique sub elements",                estr(self.unique_sub_elements))))
