"""FormData class easy for collecting of various data about a form."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-13 -- 2008-12-08"

# Modified by Anders Logg, 2008

from itertools import chain

from ufl.output import ufl_assert
from ufl.common import lstr, tstr, domain2dim
from ufl.form import Form

from ufl.algorithms.analysis import extract_basisfunctions, extract_coefficients, extract_classes

class FormData(object):
    "Class collecting various information extracted from a Form."
    
    def __init__(self, form, name="a", coefficient_names=None):
        "Create form data for given form"
        ufl_assert(isinstance(form, Form), "Expecting Form.")
        
        self.form = form
        self.name = name

        # Get arguments and their elements
        self.basisfunctions  = extract_basisfunctions(form)
        self.coefficients    = extract_coefficients(form)
        self.elements        = [f._element for f in chain(self.basisfunctions, self.coefficients)]
        self.unique_elements = set(self.elements)
        self.domain          = self.elements[0].domain()
        
        # Some useful dimensions
        self.rank = len(self.basisfunctions)
        self.num_coefficients = len(self.coefficients)
        self.geometric_dimension = domain2dim[self.domain]
        self.topological_dimension = self.geometric_dimension
        
        # Set coefficient names to default if necessary
        if coefficient_names is None:
            self.coefficient_names = ["w%d" % i for i in range(self.num_coefficients)]
        else:
            self.coefficient_names = coefficient_names
        
        # Build renumbering of arguments, since Function and BasisFunction
        # count doesn't necessarily match their exact order in the argument list
        def argument_renumbering(arguments):
            return dict((f,k) for (k,f) in enumerate(arguments))
        self.basisfunction_renumbering = argument_renumbering(self.basisfunctions)
        self.coefficient_renumbering = argument_renumbering(self.coefficients)
        
        # The set of all UFL classes used in each integral,
        # can be used to easily check for unsupported operations
        self.classes = {}
        for i in form.cell_integrals():
            self.classes[i] = extract_classes(i._integrand)
        for i in form.exterior_facet_integrals():
            self.classes[i] = extract_classes(i._integrand)
        for i in form.interior_facet_integrals():
            self.classes[i] = extract_classes(i._integrand)

    def __str__(self):
        "Return formatted summary of form data"
        return tstr((("Name",                     self.name),
                     ("Domain",                   self.domain),
                     ("Geometric dimension",      self.geometric_dimension),
                     ("Topological dimension",    self.topological_dimension),
                     ("Rank",                     self.rank),
                     ("Number of coefficients",   self.num_coefficients),
                     ("Number of cell integrals", len(self.form.cell_integrals())),
                     ("Number of exterior facet integrals", len(self.form.exterior_facet_integrals())),
                     ("Number of interior facet integrals", len(self.form.interior_facet_integrals())),
                     ("Basis functions",          lstr(self.basisfunctions)),
                     ("Coefficients",             lstr(self.coefficients)),
                     ("Coefficient names",        lstr(self.coefficient_names)),
                     ("Unique elements",          lstr(self.unique_elements))))
