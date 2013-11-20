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
from itertools import chain
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.integral import Integral, Measure, is_scalar_constant_expression
from ufl.equation import Equation
from ufl.expr import Expr
from ufl.constantvalue import Zero


# --- The Form class, representing a complete variational form or functional ---

def integral_sequence_to_dict(integrals):
    "Map a sequence of Integral objects to a dictionary of lists of Integrals keyed by domain type."
    d = {}
    for itg in integrals:
        k = itg.domain_type()
        if k not in d:
            d[k] = [itg]
        else:
            d[k].append(itg)
    return d

def integral_dict_to_sequence(integrals):
    "Map a dictionary of lists of Integrals keyed by domain type into a sequence of Integral objects ."
    return tuple(itg for dt in Measure._domain_types_tuple
                 for itg in integrals.get(dt, ()))

def join_dintegrals(aintegrals, bintegrals):
    # Store integrals from two forms in a canonical sorting
    return integral_sequence_to_dict(chain(integral_dict_to_sequence(aintegrals),
                                           integral_dict_to_sequence(bintegrals)))

class Form(object):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = ("_dintegrals",      # Dict of one integral list per domain type
                 "_hash",            # Hash code for use in dicts, including incidental numbering of indices etc.
                 "_signature",       # Signature for use with jit cache, independent of incidental numbering of indices etc.
                 "_form_data",       # Cache of preprocess result applied to this form
                 "_is_preprocessed", # Set to true if this form is the result of a preprocess of another form
                 #"_domain_data",    # TODO: Make domain data a property of the form instead of the integral?
                 )

    def __init__(self, integrals):
        self._dintegrals = integral_sequence_to_dict(integrals)
        ufl_assert(all(isinstance(itg, Integral) for itg in integrals),
                   "Expecting list of integrals.")
        self._signature = None
        self._hash = None
        self._form_data = None
        self._is_preprocessed = False

    def cell(self): # TODO: DEPRECATE
        #from ufl.log import deprecate
        #deprecate("Form.cell is not well defined and will be removed.")
        for itg in self.integrals():
            cell = itg.integrand().cell()
            if cell is not None:
                return cell
        return None

    def integral_groups(self):
        """Return a dict, which is a mapping from domain types to integrals.

        In particular, the keys are domain_type strings, and the
        values are lists of Integral instances. The Integrals in
        each list share the same domain type (the key), but have
        potentially different domain ids and metadata."""
        return self._dintegrals

    def integrals(self, domain_type=None):
        if domain_type is None:
            return integral_dict_to_sequence(self.integral_groups())
        else:
            return self.integral_groups().get(domain_type,[])

    def is_preprocessed(self):
        "Return true if this form is the result of a preprocessing of another form."
        return self._is_preprocessed

    def form_data(self):
        "Return form metadata (None if form has not been preprocessed)"
        return self._form_data

    def compute_form_data(self,
                          object_names=None,
                          common_cell=None,
                          element_mapping=None):
        "Compute and return form metadata"
        # TODO: We should get rid of the form data caching, but need to
        #       figure out how to do that and keep pydolfin working properly

        # Only compute form data once, and never on an already processed form
        ufl_assert(not self.is_preprocessed(), "You can not preprocess forms twice.")
        if self._form_data is None:
            from ufl.algorithms.preprocess import preprocess
            self._form_data = preprocess(self,
                                         object_names=object_names,
                                         common_cell=common_cell,
                                         element_mapping=element_mapping)
        # Always validate arguments, keeping sure that the validation works
        self._form_data.validate(object_names=object_names,
                                 common_cell=common_cell,
                                 element_mapping=element_mapping)
        return self.form_data()

    def __eq__(self, other):
        return Equation(self, other)

    def __radd__(self, other):
        # Ordering of form additions make no difference
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Form):
            # Add integrands of integrals with the same measure
            return Form(integral_dict_to_sequence(join_dintegrals(self.integral_groups(), other.integral_groups())))
        elif isinstance(other, (int,float)) and other == 0:
            # Allow adding 0 or 0.0 as a no-op, needed for sum([a,b])
            return self
        elif isinstance(other, Zero) and not (other.shape() or other.free_indices()):
            # Allow adding ufl Zero as a no-op, needed for sum([a,b])
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
        s = "\n  +  ".join(str(itg) for itg in self.integrals())
        return s or "<empty Form>"

    def __repr__(self):
        # TODO: Add warning here to check if anyone actually calls it in libraries
        # Not caching this because it can be huge
        r = "Form([%s])" % ", ".join(repr(itg) for itg in self.integrals())
        return r

    def __hash__(self):
        if self._hash is None:
            hashdata = tuple(hash(itg) for itg in self.integrals())
            self._hash = hash(hashdata)
        return self._hash

    def deprecated_signature(self, function_replace_map=None):
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
