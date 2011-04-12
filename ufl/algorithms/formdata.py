"""FormData class easy for collecting of various data about a form."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-09-13"

# Modified by Anders Logg, 2008.
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
