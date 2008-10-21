"""FormData class easy for collecting of various data about a form."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-13 -- 2008-10-21"

# Modified by Anders Logg, 2008

from ..output import ufl_assert
from ..common import lstr, tstr, domain_to_dim
from ..form import Form

# TODO: FormData can be constructed more efficiently as a single or a few algorithms.
from .analysis import extract_basisfunctions, extract_coefficients, extract_classes
from .analysis import extract_elements, extract_unique_elements, extract_domain, extract_classes

class FormData(object):
    "Class collecting various information extracted from form."
    
    def __init__(self, form):
        "Create form data for given form"
        ufl_assert(isinstance(form, Form), "Expecting Form.")
        
        self.form = form
        
        self.basisfunctions  = extract_basisfunctions(form)
        self.coefficients    = extract_coefficients(form)
        self.elements        = extract_elements(form)
        self.unique_elements = extract_unique_elements(form)
        self.domain          = extract_domain(form)
        
        self.rank = len(self.basisfunctions)
        self.num_coefficients = len(self.coefficients)
        self.geometric_dimension = domain_to_dim(self.domain)
        self.topological_dimension = self.geometric_dimension
        
        def argument_renumbering(arguments):
            return dict((f,k) for (k,f) in enumerate(arguments))
        self.basisfunction_renumbering = argument_renumbering(self.basisfunctions)
        self.coefficient_renumbering = argument_renumbering(self.coefficients)
        
        self.classes = {}
        for i in form.cell_integrals():
            self.classes[i] = classes(i._integrand)
        for i in form.exterior_facet_integrals():
            self.classes[i] = classes(i._integrand)
        for i in form.interior_facet_integrals():
            self.classes[i] = classes(i._integrand)

    def __str__(self):
        "Print summary of form data"

        return tstr((("Domain",                   self.domain),
                     ("Geometric dimension",      self.geometric_dimension),
                     ("Topological dimension",    self.topological_dimension),
                     ("Rank",                     self.rank),
                     ("Number of coefficients",   self.num_coefficients),
                     ("Number of cell integrals", len(self.form.cell_integrals())),
                     ("Number of e.f. integrals", len(self.form.exterior_facet_integrals())),
                     ("Number of i.f. integrals", len(self.form.interior_facet_integrals())),
                     ("Basis functions",          lstr(self.basisfunctions)),
                     ("Coefficients",             lstr(self.coefficients)),
                     ("Unique elements",          lstr(self.unique_elements))))
