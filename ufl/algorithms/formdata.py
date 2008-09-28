"""FormData class easy for collecting of various data about a form."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-13 -- 2008-09-18"

# Modified by Anders Logg, 2008

from ..output import ufl_assert
from ..form import Form

# TODO: FormData can be constructed more efficiently as a single or a few algorithms.
from .analysis import basisfunctions, coefficients, elements, unique_elements, domain, classes

class FormData(object):
    "Class collecting various information extracted from form."
    
    def __init__(self, form):
        "Create form data for given form"
        ufl_assert(isinstance(form, Form), "Expecting Form.")

        self.form = form
        
        self.basisfunctions  = basisfunctions(form)
        self.coefficients    = coefficients(form)
        self.elements        = elements(form)
        self.unique_elements = unique_elements(form)
        self.domain          = domain(form)
        
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

        def lstr(l):
            "Pretty-print list, invoking str() on items instead of repr() like str(list) does."
            return "[" + ", ".join([str(item) for item in l]) + "]"

        return """\
Basis functions: %s
Coefficients:    %s
Finite elements: %s
Unique elements: %s
Domain:          %s
Renumbering (v): %s
Renumbering (w): %s
Classes:         %s
""" % (lstr(self.basisfunctions),
       lstr(self.coefficients),
       lstr(self.elements),
       lstr(self.unique_elements),
       str(self.domain),
       lstr(self.basisfunction_renumbering),
       lstr(self.coefficient_renumbering),
       lstr(self.classes))
