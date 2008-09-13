"""FormData class easy for collecting of various data about a form."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-13 -- 2008-09-13"

from ..output import ufl_assert
from ..form import Form


# TODO: This can be implemented more efficiently as a single alogorithm.

class FormData(object):
    "Class collecting various information extracted from form."
    def __init__(self, form):
        ufl_assert(isinstance(form, Form), "Expecting Form.")
        self.form = form
        
        self.basisfunctions  = basisfunctions(form)
        self.coefficients    = coefficients(form)
        self.elements        = elements(form)
        self.unique_elements = unique_elements(form)
        
        self.domain          = domain(form)
        
        self.classes         = {}
        for i in form.cell_integrals():
            self.classes[i] = classes(i._integrand)
        for i in form.exterior_facet_integrals():
            self.classes[i] = classes(i._integrand)
        for i in form.interior_facet_integrals():
            self.classes[i] = classes(i._integrand)

def build_form_data(form):
    "Build an object with various data extracted from the given Form object."
    return FormData(form)

