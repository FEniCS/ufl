"""FormData class easy for collecting of various data about a form."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-13 -- 2009-04-25"

# Modified by Anders Logg, 2008.

from itertools import chain

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.common import lstr, tstr, sstr
from ufl.form import Form

from ufl.algorithms.analysis import extract_arguments, extract_sub_elements, build_argument_replace_map
from ufl.algorithms.transformations import replace

from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.renumbering import renumber_indices


def estr(elements):
    return ", ".join(e.shortstr() for e in elements)

class FormData(object):
    "Class collecting various information extracted from a Form."
    
    def __init__(self, form, name="a"):
        "Create form data for given form"
        ufl_assert(isinstance(form, Form), "Expecting Form.")
        
        self.name = name
        self.original_form = form
        del form # to avoid bugs
        
        # Expanding all derivatives. This (currently) also has
        # the side effect that compounds are expanded.
        # TODO: Should we really apply this here?
        # This was a convenient place to put it for temporary testing,
        # but we should discuss global application data flow!
        # One reason for putting it here is that functional derivatives
        # may change the number of form arguments, which is critical
        # for the rest of this function.
        self.form = expand_derivatives(self.original_form)
        
        if not self.form._integrals:
            error("Form is empty after transformations, can't extract form data.")

        # Renumber indices to start from 0, as a simple attempt at making
        # the form signature (repr) consistent independent of when in the
        # application a form is created. This is not foolproof, but better
        # than nothing.
        self.form = renumber_indices(self.form)

        # Get arguments and their elements
        basis_functions, functions = extract_arguments(self.form)

        # Replace arguments with new objects renumbered with count internal to the form
        replace_map, self.basis_functions, self.functions = \
            build_argument_replace_map(basis_functions, functions)
        self.form = replace(self.form, replace_map)
        del basis_functions # debugging, to avoid bugs below
        del functions # debugging, to avoid bugs below

        # Build mapping from new form argument objects to the
        # original form argument objects, in case the original
        # objects had external data attached to them
        # (PyDOLFIN does that)
        original_arguments = {}
        for k,v in replace_map.iteritems():
            original_arguments[v] = k
        self.original_basis_functions = [original_arguments[f] for f in self.basis_functions]
        self.original_functions = [original_arguments[f] for f in self.functions]
        del original_arguments # debugging, to avoid bugs below

        # Some useful dimensions
        self.rank = len(self.basis_functions)
        self.num_functions = len(self.functions)

        # Define default function names
        self.function_names = ["w%d" % i for i in range(self.num_functions)]
        self.basis_function_names = ["v%d" % i for i in range(self.rank)]

        # Get all elements
        self.elements = [f._element for f in chain(self.basis_functions, self.functions)]

        # Make a set of all unique top-level elements
        self.unique_elements = set(self.elements)

        # Make a set of all unique elements
        self.sub_elements = set(chain(*[extract_sub_elements(sub) for sub in self.unique_elements]))

        # Get geometric information
        if self.elements:
            self.cell = self.elements[0].cell()
        else:
            # Special case to allow functionals only depending on geometric variables, with no elements
            self.cell = self.form._integrals[0].integrand().cell()
        self.geometric_dimension = self.cell.d
        self.topological_dimension = self.geometric_dimension
        
        # Attach form data to both original form and transformed form,
        # to ensure the invariant "form_data.form.form_data() is form_data"
        self.original_form._form_data = self
        self.form._form_data = self

    def __str__(self):
        "Return formatted summary of form data"
        return tstr((("Name",                               self.name),
                     ("Rank",                               self.rank),                     
                     ("Cell",                               self.cell),
                     ("Geometric dimension",                self.geometric_dimension),
                     ("Topological dimension",              self.topological_dimension),
                     ("Number of functions",                self.num_functions),
                     ("Number of cell integrals",           len(self.form.cell_integrals())),
                     ("Number of exterior facet integrals", len(self.form.exterior_facet_integrals())),
                     ("Number of interior facet integrals", len(self.form.interior_facet_integrals())),
                     ("Basis functions",                    lstr(self.basis_functions)),
                     ("Functions",                          lstr(self.functions)),
                     ("Basis function names",               lstr(self.function_names)),
                     ("Function names",                     lstr(self.function_names)),
                     ("Unique elements",                    estr(self.unique_elements)),
                     ("Unique sub elements",                estr(self.sub_elements)),
                    ))
