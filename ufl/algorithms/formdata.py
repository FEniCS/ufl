"""FormData class easy for collecting of various data about a form."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-09-13 -- 2009-02-25"

# Modified by Anders Logg, 2008.

from itertools import chain

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
    
    def __init__(self, form, name="a", coefficient_names=None):
        "Create form data for given form"
        ufl_assert(isinstance(form, Form), "Expecting Form.")
        
        self.name = name
        self.original_form = form
        del form # to avoid bugs
        
        # Expanding all derivatives. This (currently) also has
        # the side effect that compounds are expanded.
        # FIXME: Should we really apply this here?
        # This was a convenient place to put it for temporary testing,
        # but we should discuss global application data flow!
        # One reason for putting it here is that functional derivatives
        # may change the number of form arguments, which is critical
        # for the rest of this function.
        self.form = expand_derivatives(self.original_form)
        
        # Renumber indices to start from 0, as a simple attempt at making
        # the form signature (repr) consistent independent of when in the
        # application a form is created. This is not foolproof, but better
        # than nothing.
        self.form = renumber_indices(self.form)

        # Get arguments and their elements
        basis_functions, coefficients = extract_arguments(self.form)

        # Replace arguments with new objects renumbered with count internal to the form
        replace_map, self.basis_functions, self.coefficients = \
            build_argument_replace_map(basis_functions, coefficients)
        self.form = replace(self.form, replace_map)
        del basis_functions # to avoid bugs
        del coefficients # to avoid bugs

        # Build mapping from new form argument objects to the
        # original form argument objects, in case the original
        # objects had external data attached to them
        # (PyDOLFIN does that)
        self.original_arguments = {}
        for k,v in replace_map.iteritems():
            self.original_arguments[v] = k

        # Some useful dimensions
        self.rank = len(self.basis_functions)
        self.num_coefficients = len(self.coefficients)

        # Set coefficient names to default if necessary
        if coefficient_names is None:
            self.coefficient_names = ["w%d" % i for i in range(self.num_coefficients)]
        else:
            self.coefficient_names = coefficient_names

        # Get all elements
        self.elements = [f._element for f in chain(self.basis_functions, self.coefficients)]

        # Make a set of all unique top-level elements
        self.unique_elements = set(self.elements)

        # Make a set of all unique elements
        self.sub_elements = set(chain(*[extract_sub_elements(sub) for sub in self.unique_elements]))

        # Get geometric information
        self.cell = self.elements[0].cell()
        self.geometric_dimension = self.cell.d
        self.topological_dimension = self.geometric_dimension

        # Estimate a default integration order, form compiler can overrule
        # TODO: Provide a better estimate
        quadrature_elements = [e for e in self.sub_elements if "Quadrature" in e.family()]
        if quadrature_elements:
            # Either take the order from quadrature elements among the coefficients...
            quad_order = quadrature_elements[0].degree()
            ufl_assert(all(quad_order == e.degree() for e in quadrature_elements),
                "Incompatible quadrature elements specified (orders must be equal).")
        else:
            # Use sum of basis_function degrees
            quad_order = sum(b.element().degree() for b in self.basis_functions)
        self.quad_order = quad_order
        
        # Attach form data to original form
        self.original_form._form_data = self

    def __str__(self):
        "Return formatted summary of form data"
        return tstr((("Name",                               self.name),
                     ("Cell",                               self.cell),
                     ("Geometric dimension",                self.geometric_dimension),
                     ("Topological dimension",              self.topological_dimension),
                     ("Rank",                               self.rank),
                     ("Number of coefficients",             self.num_coefficients),
                     ("Number of cell integrals",           len(self.form.cell_integrals())),
                     ("Number of exterior facet integrals", len(self.form.exterior_facet_integrals())),
                     ("Number of interior facet integrals", len(self.form.interior_facet_integrals())),
                     ("Basis functions",                    lstr(self.basis_functions)),
                     ("Coefficients",                       lstr(self.coefficients)),
                     ("Coefficient names",                  lstr(self.coefficient_names)),
                     ("Unique elements",                    estr(self.unique_elements)),
                     ("Unique sub elements",                estr(self.sub_elements)),
                     ("Estimated quadrature order",         self.quad_order),
                    ))
