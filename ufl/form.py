# -*- coding: utf-8 -*-
"The Form class."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2011.
# Modified by Massimiliano Leoni, 2016.
# Modified by Cecile Daversin-Catty, 2018.

import warnings
from collections import defaultdict
from itertools import chain

from ufl.checks import is_scalar_constant_expression
from ufl.constant import Constant
from ufl.constantvalue import Zero
from ufl.core.expr import Expr, ufl_err_str
from ufl.core.ufl_type import UFLType, ufl_type
from ufl.domain import extract_unique_domain, sort_domains
from ufl.equation import Equation
from ufl.integral import Integral
from ufl.utils.counted import Counted
from ufl.utils.sorting import sorted_by_count

# Export list for ufl.classes
__all_classes__ = ["Form", "BaseForm", "ZeroBaseForm"]

# --- The Form class, representing a complete variational form or functional ---


def _sorted_integrals(integrals):
    """Sort integrals by domain id, integral type, subdomain id
    for a more stable signature computation."""

    # Group integrals in multilevel dict by keys
    # [domain][integral_type][subdomain_id]
    integrals_dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for integral in integrals:
        d = integral.ufl_domain()
        if d is None:
            raise ValueError(
                "Each integral in a form must have a uniquely defined integration domain.")
        it = integral.integral_type()
        si = integral.subdomain_id()
        integrals_dict[d][it][si] += [integral]

    all_integrals = []

    # Order integrals canonically to increase signature stability
    for d in sort_domains(integrals_dict):
        for it in sorted(integrals_dict[d]):  # str is sortable
            for si in sorted(integrals_dict[d][it], key=lambda x: (type(x).__name__, x)):  # int/str are sortable
                unsorted_integrals = integrals_dict[d][it][si]
                # TODO: At this point we could order integrals by
                #       metadata and integrand, or even add the
                #       integrands with the same metadata. This is done
                #       in accumulate_integrands_with_same_metadata in
                #       algorithms/domain_analysis.py and would further
                #       increase the signature stability.
                all_integrals.extend(unsorted_integrals)
                # integrals_dict[d][it][si] = unsorted_integrals

    return tuple(all_integrals)  # integrals_dict


@ufl_type()
class BaseForm(object, metaclass=UFLType):
    """Description of an object containing arguments"""

    # Slots is kept empty to enable multiple inheritance with other
    # classes
    __slots__ = ()
    _ufl_is_abstract_ = True
    _ufl_required_methods_ = ('_analyze_form_arguments', "ufl_domains")

    def __init__(self):
        # Internal variables for caching form argument data
        self._arguments = None

    # --- Accessor interface ---
    def arguments(self):
        "Return all ``Argument`` objects found in form."
        if self._arguments is None:
            self._analyze_form_arguments()
        return self._arguments

    # --- Operator implementations ---

    def __eq__(self, other):
        """Delayed evaluation of the == operator!

        Just 'lhs_form == rhs_form' gives an Equation,
        while 'bool(lhs_form == rhs_form)' delegates
        to lhs_form.equals(rhs_form).
        """
        return Equation(self, other)

    def __radd__(self, other):
        # Ordering of form additions make no difference
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            # Allow adding 0 or 0.0 as a no-op, needed for sum([a,b])
            return self
        elif isinstance(other, Zero) and not (other.ufl_shape or other.ufl_free_indices):
            # Allow adding ufl Zero as a no-op, needed for sum([a,b])
            return self

        elif isinstance(other, ZeroBaseForm):
            self._check_arguments_sum(other)
            # Simplify addition with ZeroBaseForm
            return self

        # For `ZeroBaseForm(...) + B` with B a BaseForm.
        # We could overwrite ZeroBaseForm.__add__ but that implies
        # duplicating cases with `0` and `ufl.Zero`.
        elif isinstance(self, ZeroBaseForm):
            self._check_arguments_sum(other)
            # Simplify addition with ZeroBaseForm
            return other

        elif isinstance(other, BaseForm):
            # Add integrals from both forms
            return FormSum((self, 1), (other, 1))

        else:
            # Let python protocols do their job if we don't handle it
            return NotImplemented

    def _check_arguments_sum(self, other):
        # Get component with the highest number of arguments
        a = max((self, other), key=lambda x: len(x.arguments()))
        b = self if a is other else other
        # Components don't necessarily have the exact same arguments
        # but the first argument(s) need to match as for `a + L`
        # where a and L are a bilinear and linear form respectively.
        a_args = sorted(a.arguments(), key=lambda x: x.number())
        b_args = sorted(b.arguments(), key=lambda x: x.number())
        if b_args != a_args[:len(b_args)]:
            raise ValueError('Mismatching arguments when summing:\n %s\n and\n %s' % (self, other))

    def __sub__(self, other):
        "Subtract other form from this one."
        return self + (-other)

    def __rsub__(self, other):
        "Subtract this form from other."
        return other + (-self)

    def __neg__(self):
        """Negate all integrals in form.

        This enables the handy "-form" syntax for e.g. the
        linearized system (J, -F) from a nonlinear form F."""
        if isinstance(self, ZeroBaseForm):
            # `-` doesn't change anything for ZeroBaseForm.
            # This also facilitates simplifying FormSum containing ZeroBaseForm objects.
            return self
        return FormSum((self, -1))

    def __rmul__(self, scalar):
        "Multiply all integrals in form with constant scalar value."
        # This enables the handy "0*form" or "dt*form" syntax
        if is_scalar_constant_expression(scalar):
            return FormSum((self, scalar))
        return NotImplemented

    def __mul__(self, coefficient):
        "Take the action of this form on the given coefficient."
        if isinstance(coefficient, Expr):
            from ufl.formoperators import action
            return action(self, coefficient)
        return NotImplemented

    def __ne__(self, other):
        "Immediately evaluate the != operator (as opposed to the == operator)."
        return not self.equals(other)

    def __call__(self, *args, **kwargs):
        """Evaluate form by replacing arguments and coefficients.

        Replaces form.arguments() with given positional arguments in
        same number and ordering. Number of positional arguments must
        be 0 or equal to the number of Arguments in the form.

        The optional keyword argument coefficients can be set to a dict
        to replace Coefficients with expressions of matching shapes.

        Example:
        -------
          V = FiniteElement("CG", triangle, 1)
          v = TestFunction(V)
          u = TrialFunction(V)
          f = Coefficient(V)
          g = Coefficient(V)
          a = g*inner(grad(u), grad(v))*dx
          M = a(f, f, coefficients={ g: 1 })

        Is equivalent to M == grad(f)**2*dx.

        """
        repdict = {}

        if args:
            arguments = self.arguments()
            if len(arguments) != len(args):
                raise ValueError(f"Need {len(arguments)} arguments to form(), got {len(args)}.")
            repdict.update(zip(arguments, args))

        coefficients = kwargs.pop("coefficients")
        if kwargs:
            raise ValueError(f"Unknown kwargs {list(kwargs)}.")

        if coefficients is not None:
            coeffs = self.coefficients()
            for f in coefficients:
                if f in coeffs:
                    repdict[f] = coefficients[f]
                else:
                    warnings("Coefficient %s is not in form." % ufl_err_str(f))

        if repdict:
            from ufl.formoperators import replace
            return replace(self, repdict)
        else:
            return self

    def _ufl_compute_hash_(self):
        "Compute the hash"
        # Ensure compatibility with MultiFunction
        # `hash(self)` will call the `__hash__` method of the subclass.
        return hash(self)

    def _ufl_expr_reconstruct_(self, *operands):
        "Return a new object of the same type with new operands."
        return type(self)(*operands)

    # "a @ f" notation in python 3.5
    __matmul__ = __mul__

    # --- String conversion functions, for UI purposes only ---


@ufl_type()
class Form(BaseForm):
    """Description of a weak form consisting of a sum of integrals over subdomains."""
    __slots__ = (
        # --- List of Integral objects (a Form is a sum of these Integrals, everything else is derived)
        "_integrals",
        # --- Internal variables for caching various data
        "_integration_domains",
        "_domain_numbering",
        "_subdomain_data",
        "_arguments",
        "_coefficients",
        "_coefficient_numbering",
        "_constants",
        "_constant_numbering",
        "_terminal_numbering",
        "_hash",
        "_signature",
        # --- Dict that external frameworks can place framework-specific
        #     data in to be carried with the form
        #     Never use this internally in ufl!
        "_cache",
    )

    def __init__(self, integrals):
        BaseForm.__init__(self)
        # Basic input checking (further compatibilty analysis happens
        # later)
        if not all(isinstance(itg, Integral) for itg in integrals):
            raise ValueError("Expecting list of integrals.")

        # Store integrals sorted canonically to increase signature
        # stability
        self._integrals = _sorted_integrals(integrals)

        # Internal variables for caching domain data
        self._integration_domains = None
        self._domain_numbering = None

        # Internal variables for caching subdomain data
        self._subdomain_data = None

        # Internal variables for caching form argument data
        self._coefficients = None
        self._coefficient_numbering = None
        self._constant_numbering = None
        self._terminal_numbering = None

        from ufl.algorithms.analysis import extract_constants
        self._constants = extract_constants(self)

        # Internal variables for caching of hash and signature after
        # first request
        self._hash = None
        self._signature = None

        # Never use this internally in ufl!
        self._cache = {}

    # --- Accessor interface ---

    def integrals(self):
        "Return a sequence of all integrals in form."
        return self._integrals

    def integrals_by_type(self, integral_type):
        "Return a sequence of all integrals with a particular domain type."
        return tuple(integral for integral in self.integrals()
                     if integral.integral_type() == integral_type)

    def integrals_by_domain(self, domain):
        "Return a sequence of all integrals with a particular integration domain."
        return tuple(integral for integral in self.integrals() if integral.ufl_domain() == domain)

    def empty(self):
        "Returns whether the form has no integrals."
        return self.integrals() == ()

    def ufl_domains(self):
        """Return the geometric integration domains occuring in the form.

        NB! This does not include domains of coefficients defined on other meshes.

        The return type is a tuple even if only a single domain exists.
        """
        if self._integration_domains is None:
            self._analyze_domains()
        return self._integration_domains

    def ufl_cell(self):
        """Return the single cell this form is defined on, fails if multiple
        cells are found.

        """
        return self.ufl_domain().ufl_cell()

    def ufl_domain(self):
        """Return the single geometric integration domain occuring in the
        form.

        Fails if multiple domains are found.

        NB! This does not include domains of coefficients defined on
        other meshes, look at form data for that additional
        information.

        """
        # Collect all domains
        domains = self.ufl_domains()
        # Check that all are equal TODO: don't return more than one if
        # all are equal?
        if not all(domain == domains[0] for domain in domains):
            raise ValueError("Calling Form.ufl_domain() is only valid if all integrals share domain.")

        # Return the one and only domain
        return domains[0]

    def geometric_dimension(self):
        """Return the geometric dimension shared by all domains and functions in this form."""
        gdims = tuple(
            set(domain.geometric_dimension() for domain in self.ufl_domains()))
        if len(gdims) != 1:
            raise ValueError("Expecting all domains and functions in a form "
                             f"to share geometric dimension, got {tuple(sorted(gdims))}")
        return gdims[0]

    def domain_numbering(self):
        """Return a contiguous numbering of domains in a mapping
        ``{domain:number}``."""
        if self._domain_numbering is None:
            self._analyze_domains()
        return self._domain_numbering

    def subdomain_data(self):
        """Returns a mapping on the form ``{domain:{integral_type:
            subdomain_data}}``."""
        if self._subdomain_data is None:
            self._analyze_subdomain_data()
        return self._subdomain_data

    def max_subdomain_ids(self):
        """Returns a mapping on the form
        ``{domain:{integral_type:max_subdomain_id}}``."""
        if self._max_subdomain_ids is None:
            self._analyze_subdomain_data()
        return self._max_subdomain_ids

    def arguments(self):
        "Return all ``Argument`` objects found in form."
        if self._arguments is None:
            self._analyze_form_arguments()
        return self._arguments

    def coefficients(self):
        "Return all ``Coefficient`` objects found in form."
        if self._coefficients is None:
            self._analyze_form_arguments()
        return self._coefficients

    def coefficient_numbering(self):
        """Return a contiguous numbering of coefficients in a mapping
        ``{coefficient:number}``."""
        # cyclic import
        from ufl.coefficient import Coefficient

        if self._coefficient_numbering is None:
            self._coefficient_numbering = {
                expr: num
                for expr, num in self.terminal_numbering().items()
                if isinstance(expr, Coefficient)
            }
        return self._coefficient_numbering

    def constants(self):
        return self._constants

    def constant_numbering(self):
        """Return a contiguous numbering of constants in a mapping
        ``{constant:number}``."""
        if self._constant_numbering is None:
            self._constant_numbering = {
                expr: num
                for expr, num in self.terminal_numbering().items()
                if isinstance(expr, Constant)
            }
        return self._constant_numbering

    def terminal_numbering(self):
        """Return a contiguous numbering for all counted objects in the form.

        The returned object is mapping from terminal to its number (an integer).

        The numbering is computed per type so :class:`Coefficient`s,
        :class:`Constant`s, etc will each be numbered from zero.

        """
        # cyclic import
        from ufl.algorithms.analysis import extract_type

        if self._terminal_numbering is None:
            exprs_by_type = defaultdict(set)
            for counted_expr in extract_type(self, Counted):
                exprs_by_type[counted_expr._counted_class].add(counted_expr)

            numbering = {}
            for exprs in exprs_by_type.values():
                for i, expr in enumerate(sorted_by_count(exprs)):
                    numbering[expr] = i
            self._terminal_numbering = numbering
        return self._terminal_numbering

    def signature(self):
        "Signature for use with jit cache (independent of incidental numbering of indices etc.)"
        if self._signature is None:
            self._compute_signature()
        return self._signature

    # --- Operator implementations ---

    def __hash__(self):
        "Hash code for use in dicts (includes incidental numbering of indices etc.)"
        if self._hash is None:
            self._hash = hash(tuple(hash(itg) for itg in self.integrals()))
        return self._hash

    def __ne__(self, other):
        "Immediate evaluation of the != operator (as opposed to the == operator)."
        return not self.equals(other)

    def equals(self, other):
        "Evaluate ``bool(lhs_form == rhs_form)``."
        if type(other) != Form:
            return False
        if len(self._integrals) != len(other._integrals):
            return False
        if hash(self) != hash(other):
            return False
        return all(a == b for a, b in zip(self._integrals, other._integrals))

    def __radd__(self, other):
        # Ordering of form additions make no difference
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Form):
            # Add integrals from both forms
            return Form(list(chain(self.integrals(), other.integrals())))

        if isinstance(other, ZeroBaseForm):
            self._check_arguments_sum(other)
            # Simplify addition with ZeroBaseForm
            return self

        elif isinstance(other, BaseForm):
            # Create form sum if form is of other type
            return FormSum((self, 1), (other, 1))

        elif isinstance(other, (int, float)) and other == 0:
            # Allow adding 0 or 0.0 as a no-op, needed for sum([a,b])
            return self

        elif isinstance(
                other,
                Zero) and not (other.ufl_shape or other.ufl_free_indices):
            # Allow adding ufl Zero as a no-op, needed for sum([a,b])
            return self

        else:
            # Let python protocols do their job if we don't handle it
            return NotImplemented

    def __sub__(self, other):
        "Subtract other form from this one."
        return self + (-other)

    def __rsub__(self, other):
        "Subtract this form from other."
        return other + (-self)

    def __neg__(self):
        """Negate all integrals in form.

        This enables the handy "-form" syntax for e.g. the
        linearized system (J, -F) from a nonlinear form F."""
        return Form([-itg for itg in self.integrals()])

    def __rmul__(self, scalar):
        "Multiply all integrals in form with constant scalar value."
        # This enables the handy "0*form" or "dt*form" syntax
        if is_scalar_constant_expression(scalar):
            return Form([scalar * itg for itg in self.integrals()])
        return NotImplemented

    def __mul__(self, coefficient):
        "UFL form operator: Take the action of this form on the given coefficient."
        if isinstance(coefficient, Expr):
            from ufl.formoperators import action
            return action(self, coefficient)
        return NotImplemented

    def __call__(self, *args, **kwargs):
        """UFL form operator: Evaluate form by replacing arguments and
        coefficients.

        Replaces form.arguments() with given positional arguments in
        same number and ordering. Number of positional arguments must
        be 0 or equal to the number of Arguments in the form.

        The optional keyword argument coefficients can be set to a dict
        to replace Coefficients with expressions of matching shapes.

        Example:
        -------
          V = FiniteElement("CG", triangle, 1)
          v = TestFunction(V)
          u = TrialFunction(V)
          f = Coefficient(V)
          g = Coefficient(V)
          a = g*inner(grad(u), grad(v))*dx
          M = a(f, f, coefficients={ g: 1 })

        Is equivalent to M == grad(f)**2*dx.

        """
        repdict = {}

        if args:
            arguments = self.arguments()
            if len(arguments) != len(args):
                raise ValueError(f"Need {len(arguments)} arguments to form(), got {len(args)}.")
            repdict.update(zip(arguments, args))

        coefficients = kwargs.pop("coefficients", None)
        if kwargs:
            raise ValueError(f"Unknown kwargs {list(kwargs)}")

        if coefficients is not None:
            coeffs = self.coefficients()
            for f in coefficients:
                if f in coeffs:
                    repdict[f] = coefficients[f]
                else:
                    warnings.warn("Coefficient %s is not in form." % ufl_err_str(f))
        if repdict:
            from ufl.formoperators import replace
            return replace(self, repdict)
        else:
            return self

    # "a @ f" notation in python 3.5
    __matmul__ = __mul__

    # --- String conversion functions, for UI purposes only ---

    def __str__(self):
        "Compute shorter string representation of form. This can be huge for complicated forms."
        # Warning used for making sure we don't use this in the general pipeline:
        # warning("Calling str on form is potentially expensive and should be avoided except during debugging.")
        # Not caching this because it can be huge
        s = "\n  +  ".join(str(itg) for itg in self.integrals())
        return s or "<empty Form>"

    def __repr__(self):
        "Compute repr string of form. This can be huge for complicated forms."
        # Warning used for making sure we don't use this in the general pipeline:
        # warning("Calling repr on form is potentially expensive and should be avoided except during debugging.")
        # Not caching this because it can be huge
        itgs = ", ".join(repr(itg) for itg in self.integrals())
        r = "Form([" + itgs + "])"
        return r

    # --- Analysis functions, precomputation and caching of various quantities

    def _analyze_domains(self):
        from ufl.domain import join_domains, sort_domains

        # Collect unique integration domains
        integration_domains = join_domains([itg.ufl_domain() for itg in self._integrals])

        # Make canonically ordered list of the domains
        self._integration_domains = sort_domains(integration_domains)

        # TODO: Not including domains from coefficients and arguments
        # here, may need that later
        self._domain_numbering = dict((d, i) for i, d in enumerate(self._integration_domains))

    def _analyze_subdomain_data(self):
        integration_domains = self.ufl_domains()
        integrals = self.integrals()

        # Make clear data structures to collect subdomain data in
        subdomain_data = {}
        for domain in integration_domains:
            subdomain_data[domain] = {}

        for integral in integrals:
            # Get integral properties
            domain = integral.ufl_domain()
            it = integral.integral_type()
            sd = integral.subdomain_data()

            # Collect subdomain data
            if subdomain_data[domain].get(it) is None:
                subdomain_data[domain][it] = [sd]
            else:
                subdomain_data[domain][it].append(sd)
        self._subdomain_data = subdomain_data

    def _analyze_form_arguments(self):
        "Analyze which Argument and Coefficient objects can be found in the form."
        from ufl.algorithms.analysis import extract_arguments_and_coefficients
        arguments, coefficients = extract_arguments_and_coefficients(self)

        # Define canonical numbering of arguments and coefficients
        self._arguments = tuple(
            sorted(set(arguments), key=lambda x: x.number()))
        self._coefficients = tuple(
            sorted(set(coefficients), key=lambda x: x.count()))

    def _compute_renumbering(self):
        # Include integration domains and coefficients in renumbering
        dn = self.domain_numbering()
        tn = self.terminal_numbering()
        renumbering = {}
        renumbering.update(dn)
        renumbering.update(tn)

        # Add domains of coefficients, these may include domains not
        # among integration domains
        k = len(dn)
        for c in self.coefficients():
            d = extract_unique_domain(c)
            if d is not None and d not in renumbering:
                renumbering[d] = k
                k += 1

        # Add domains of arguments, these may include domains not
        # among integration domains
        for a in self._arguments:
            d = a.ufl_function_space().ufl_domain()
            if d is not None and d not in renumbering:
                renumbering[d] = k
                k += 1

        # Add domains of constants, these may include domains not
        # among integration domains
        for c in self._constants:
            d = extract_unique_domain(c)
            if d is not None and d not in renumbering:
                renumbering[d] = k
                k += 1

        return renumbering

    def _compute_signature(self):
        from ufl.algorithms.signature import compute_form_signature
        self._signature = compute_form_signature(self, self._compute_renumbering())


def as_form(form):
    "Convert to form if not a form, otherwise return form."
    if not isinstance(form, BaseForm):
        raise ValueError(f"Unable to convert object to a UFL form: {ufl_err_str(form)}")
    return form


@ufl_type()
class FormSum(BaseForm):
    """Description of a weighted sum of variational forms and form-like objects
    components is the list of Forms to be summed
    arg_weights is a list of tuples of component index and weight"""

    __slots__ = ("_arguments",
                 "_weights",
                 "_components",
                 "ufl_operands",
                 "_domains",
                 "_domain_numbering",
                 "_hash")
    _ufl_required_methods_ = ('_analyze_form_arguments')

    def __init__(self, *components):
        BaseForm.__init__(self)

        weights = []
        full_components = []
        for (component, w) in components:
            if isinstance(component, FormSum):
                full_components.extend(component.components())
                weights.extend(w * component.weights())
            else:
                full_components.append(component)
                weights.append(w)

        self._arguments = None
        self._domains = None
        self._domain_numbering = None
        self._hash = None
        self._weights = weights
        self._components = full_components
        self._sum_variational_components()
        self.ufl_operands = self._components

    def components(self):
        return self._components

    def weights(self):
        return self._weights

    def _sum_variational_components(self):
        var_forms = None
        other_components = []
        new_weights = []
        for (i, component) in enumerate(self._components):
            if isinstance(component, Form):
                if var_forms:
                    var_forms = var_forms + (self._weights[i] * component)
                else:
                    var_forms = self._weights[i] * component
            else:
                other_components.append(component)
                new_weights.append(self._weights[i])
        if var_forms:
            other_components.insert(0, var_forms)
            new_weights.insert(0, 1)
        self._components = other_components
        self._weights = new_weights

    def _analyze_form_arguments(self):
        "Return all ``Argument`` objects found in form."
        arguments = []
        for component in self._components:
            arguments.extend(component.arguments())
        self._arguments = tuple(set(arguments))

    def __hash__(self):
        "Hash code for use in dicts (includes incidental numbering of indices etc.)"
        if self._hash is None:
            self._hash = hash(tuple(hash(component) for component in self.components()))
        return self._hash

    def equals(self, other):
        "Evaluate ``bool(lhs_form == rhs_form)``."
        if type(other) != FormSum:
            return False
        if self is other:
            return True
        return (len(self.components()) == len(other.components()) and  # noqa: W504
                all(a == b for a, b in zip(self.components(), other.components())))

    def __str__(self):
        "Compute shorter string representation of form. This can be huge for complicated forms."
        # Warning used for making sure we don't use this in the general pipeline:
        # warning("Calling str on form is potentially expensive and should be avoided except during debugging.")
        # Not caching this because it can be huge
        s = "\n  +  ".join(str(component) for component in self.components())
        return s or "<empty FormSum>"

    def __repr__(self):
        "Compute repr string of form. This can be huge for complicated forms."
        # Warning used for making sure we don't use this in the general pipeline:
        # warning("Calling repr on form is potentially expensive and should be avoided except during debugging.")
        # Not caching this because it can be huge
        itgs = ", ".join(repr(component) for component in self.components())
        r = "FormSum([" + itgs + "])"
        return r


@ufl_type()
class ZeroBaseForm(BaseForm):
    """Description of a zero base form.

    ZeroBaseForm is idempotent with respect to assembly and is mostly
    used for sake of simplifying base-form expressions.

    """

    __slots__ = ("_arguments",
                 "_coefficients",
                 "ufl_operands",
                 "_hash",
                 # Pyadjoint compatibility
                 "form")

    def __init__(self, arguments):
        BaseForm.__init__(self)
        self._arguments = arguments
        self.ufl_operands = arguments
        self._hash = None
        self.form = None

    def _analyze_form_arguments(self):
        return self._arguments

    def __ne__(self, other):
        # Overwrite BaseForm.__neq__ which relies on `equals`
        return not self == other

    def __eq__(self, other):
        if type(other) is ZeroBaseForm:
            if self is other:
                return True
            return (self._arguments == other._arguments)
        elif isinstance(other, (int, float)):
            return other == 0
        else:
            return False

    def __str__(self):
        return "ZeroBaseForm(%s)" % (", ".join(str(arg) for arg in self._arguments))

    def __repr__(self):
        return "ZeroBaseForm(%s)" % (", ".join(repr(arg) for arg in self._arguments))

    def __hash__(self):
        """Hash code for use in dicts."""
        if self._hash is None:
            self._hash = hash(("ZeroBaseForm", hash(self._arguments)))
        return self._hash
