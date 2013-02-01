"The Form class."

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Modified by Anders Logg, 2009-2011.
#
# First added:  2008-03-14
# Last changed: 2013-01-02

import hashlib
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.integral import Integral, Measure, is_scalar_constant_expression
from ufl.equation import Equation
from ufl.expr import Expr


# --- The Form class, representing a complete variational form or functional ---

def dict_sum(items):
    "Construct a dict, in between dict(items) and sum(items), by accumulating items for each key."
    d = {}
    for k, v in items:
        if k not in d:
            d[k] = v
        else:
            d[k] += v
    return d

def integral_sequence_to_dict(integrals):
    "Map a sequence of Integral objects to a dictionary of lists of Integrals keyed by domain type."
    return dict_sum((itg.domain_type(), [itg]) for itg in integrals)

def integral_dict_to_sequence(integrals):
    "Map a dictionary of lists of Integrals keyed by domain type into a sequence of Integral objects ."
    return tuple(itg for dt in Measure._domain_types_tuple for itg in integrals.get(dt, ()))

def join_dintegrals_old(aintegrals, bintegrals): # Temporary implementation matching old behaviour
    return join_lintegrals(integral_dict_to_sequence(aintegrals), integral_dict_to_sequence(bintegrals))

def join_dintegrals_new(aintegrals, bintegrals): # New
    # Store integrals from two forms in a canonical sorting
    return integral_sequence_to_dict(chain(integral_dict_to_sequence(aintegrals), integral_dict_to_sequence(bintegrals)))

join_dintegrals = join_dintegrals_old # TODO: Implement and test properly

def join_lintegrals(aintegrals, bintegrals):
    newintegrals = list(aintegrals)

    # Build mapping: (measure -> newintegrals index)
    measure2idx = {}
    for i, itg in enumerate(newintegrals):
        ufl_assert(itg.measure() not in measure2idx, "Form invariant breached, found two integrals with same measure.")
        measure2idx[itg.measure()] = i

    for itg in bintegrals:
        idx = measure2idx.get(itg.measure())
        if idx is None:
            # Append integral with new measure to list
            idx = len(newintegrals)
            measure2idx[itg.measure()] = idx
            newintegrals.append(itg)
        else:
            # Accumulate integrands with same measure
            a = newintegrals[idx].integrand()
            b = itg.integrand()
            newintegrals[idx] = itg.reconstruct(a + b)

    return newintegrals

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = ("_integrals", # TODO: Deprecate this in favor of...
                 "_dintegrals", # TODO: Use this dict of integrals per domain type
                 "_hash",      # Hash code for use in dicts, including incidental numbering of indices etc.
                 "_signature", # Signature for use with jit cache, independent of incidental numbering of indices etc.
                 "_form_data",
                 "_is_preprocessed",
                 )

    def __init__(self, integrals):
        #self._integrals = tuple(integrals)
        self._dintegrals = integral_sequence_to_dict(integrals)
        self._integrals = integral_dict_to_sequence(self._dintegrals)
        ufl_assert(all(isinstance(itg, Integral) for itg in integrals),
                   "Expecting list of integrals.")
        self._signature = None
        self._hash = None
        self._form_data = None
        self._is_preprocessed = False

    def cell(self): # TODO: DEPRECATE
        #from ufl.log import deprecate
        #deprecate("Form.cell is not well defined and will be removed.")
        for itg in self._integrals:
            cell = itg.integrand().cell()
            if cell is not None:
                return cell
        return None

    def integral_groups(self):
        """Return a dict, which is a mapping from domains to integrals.

        In particular, each key of the dict is a distinct tuple
        (domain_type, domain_id), and each value is a list of
        Integral instances. The Integrals in each list share the
        same domain (the key), but have different measures."""
        return self._dintegrals

    def integrals(self, domain_type=None):
        if domain_type is None:
            return integral_dict_to_sequence(self._dintegrals)
        else:
            return self._dintegrals[domain_type]

    def measures(self, domain_type=None):
        return tuple(itg.measure() for itg in self.integrals(domain_type))

    def domains(self, domain_type=None):
        return tuple((m.domain_type(), m.domain_id()) for m in self.measures(domain_type))

    def cell_integrals(self):
        #deprecate("Please use integrals(Measure.CELL) instead.") # TODO: Deprecate this and the others to simplify Form
        return self._dintegrals.get(Measure.CELL, [])

    def exterior_facet_integrals(self):
        return self._dintegrals.get(Measure.EXTERIOR_FACET, [])

    def interior_facet_integrals(self):
        return self._dintegrals.get(Measure.INTERIOR_FACET, [])

    def point_integrals(self):
        return self._dintegrals.get(Measure.POINT, [])

    def macro_cell_integrals(self):
        return self._dintegrals.get(Measure.MACRO_CELL, [])

    def surface_integrals(self):
        return self._dintegrals.get(Measure.SURFACE, [])

    def form_data(self):
        "Return form metadata (None if form has not been preprocessed)"
        return self._form_data

    def compute_form_data(self,
                          object_names=None,
                          common_cell=None,
                          element_mapping=None,
                          replace_functions=True,
                          skip_signature=False):
        "Compute and return form metadata"
        # TODO: We should get rid of the form data caching, but need to figure out how to do that and keep pydolfin working properly
        if self._form_data is None:
            from ufl.algorithms.preprocess import preprocess
            self._form_data = preprocess(self,
                                         object_names=object_names,
                                         common_cell=common_cell,
                                         element_mapping=element_mapping,
                                         replace_functions=replace_functions,
                                         skip_signature=skip_signature)
        else:
            self._form_data.validate(object_names=object_names,
                                     common_cell=common_cell,
                                     element_mapping=element_mapping) # FIXME: Check replace_functions and skip_signature as well
        return self.form_data()

    def is_preprocessed(self):
        "Check whether form is preprocessed"
        return self._is_preprocessed

    def __eq__(self, other):
        return Equation(self, other)

    def __radd__(self, other):
        # Ordering of form additions make no difference
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Form):
            # Add integrands of integrals with the same measure
            return Form(join_dintegrals(self._dintegrals, other._dintegrals))
        elif isinstance(other, (int,float)) and other == 0:
            # Allow adding 0 or 0.0 as a no-op, needed for sum([a,b])
            return self
        else:
            # Let python protocols do their job if we don't handle it
            return NotImplemented

    def __sub__(self, other):
        "Subtract other form from this one."
        return self + (-other)

    def __neg__(self):
        """Negate all integrals in form.

        This enables the handy "-form" syntax for e.g. the
        linearized system (J, -F) from a nonlinear form F."""
        return Form([-itg for itg in self.integrals()])

    def __rmul__(self, scalar):
        "Multiply all integrals in form with constant scalar value."
        # This enables the handy "0*form" or "dt*form" syntax
        if is_scalar_constant_expression(scalar):
            return Form([scalar*itg for itg in self.integrals()])
        return NotImplemented

    def __mul__(self, coefficient):
        "UFL form operator: Take the action of this form on the given coefficient."
        if isinstance(coefficient, Expr): #Coefficient): # TODO: Check whatever makes sense
            from ufl.formoperators import action
            return action(self, coefficient)
        return NotImplemented

    def __str__(self):
        # TODO: Add warning here to check if anyone actually calls it in libraries
        if self._dintegrals:
            return "\n  +  ".join(str(itg) for itg in self.integrals())
        else:
            return "<empty Form>"

    def __repr__(self):
        # TODO: Add warning here to check if anyone actually calls it in libraries
        # Not caching this because it can be huge
        r = "Form([%s])" % ", ".join(repr(itg) for itg in self.integrals())
        return r

    def __hash__(self):
        if self._hash is None:
            hashdata = tuple((hash(itg.integrand()), hash(itg.measure()))
                             for itg in self.integrals())
            self._hash = hash(hashdata)
        return self._hash

    def signature(self, function_replace_map=None):
        if self._signature is None:
            from ufl.algorithms.signature import compute_form_signature
            self._signature = compute_form_signature(self, function_replace_map)
        assert self._signature
        return self._signature

    def x_repr_latex_(self): # TODO: This works, but enable when form latex rendering is fixed
        from ufl.algorithms import ufl2latex
        return "$$%s$$" % ufl2latex(self)

    def x_repr_png_(self): # TODO: This works, but enable when form latex rendering is fixed
        from IPython.lib.latextools import latex_to_png
        return latex_to_png(self._repr_latex_())
