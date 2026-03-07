"""FormData class easy for collecting of various data about a form."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008.
# Modified by Jørgen S. Dokken, 2026.

from functools import cached_property
from itertools import chain
from typing import Any

from ufl.algorithms.analysis import extract_coefficients, extract_sub_elements, unique_tuple
from ufl.algorithms.apply_coefficient_split import CoefficientSplitter
from ufl.algorithms.apply_restrictions import apply_restrictions, default_restriction_map
from ufl.algorithms.check_arities import check_integrand_arity
from ufl.algorithms.domain_analysis import IntegralData, reconstruct_form_from_integral_data
from ufl.algorithms.replace import replace
from ufl.classes import Argument, Coefficient, FunctionSpace, GeometricFacetQuantity
from ufl.coefficient import BaseCoefficient
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.domain import MeshSequence, extract_domains, extract_unique_domain
from ufl.form import Form, Zero
from ufl.utils.formatting import estr, lstr, tstr
from ufl.utils.sequences import max_degree


def _check_form_arity(
    integral_data: list[IntegralData], arguments: tuple[Argument], complex_mode: bool
):
    """Check if form arity is valid."""
    for integral in integral_data:
        for itg in integral.integrals:
            check_integrand_arity(itg.integrand(), arguments, complex_mode)


def _auto_select_degree(elements):
    """Automatically select degree for all elements of the form.

    This is be used in cases where the degree has not been specified by the user.
    This feature is used by DOLFIN to allow the specification of Expressions with
    undefined degrees.
    """
    # Use max degree of all elements, at least 1 (to work with
    # Lagrange elements)
    return max_degree({e.embedded_superdegree for e in elements} - {None} | {1})


def _compute_element_mapping(form: Form):
    """Compute element mapping for element replacement."""
    # The element mapping is a slightly messy concept with two use
    # cases:
    # - Expression with missing cell or element TODO: Implement proper
    #   Expression handling in UFL and get rid of this
    # - Constant with missing cell TODO: Fix anything that needs to be
    #   worked around to drop this requirement

    # Extract all elements and include subelements of mixed elements
    elements = [obj.ufl_element() for obj in chain(form.arguments(), form.coefficients())]
    elements = extract_sub_elements(elements)

    # Try to find a common degree for elements
    common_degree = _auto_select_degree(elements)

    # Compute element map
    element_mapping = {}
    for element in elements:
        # Flag for whether element needs to be reconstructed
        reconstruct = False

        # Set cell
        cell = element.cell
        if cell is None:
            domains = form.ufl_domains()
            if not all(domains[0].ufl_cell() == d.ufl_cell() for d in domains):
                raise ValueError(
                    "Cannot replace unknown element cell without unique common cell in form."
                )
            cell = domains[0].ufl_cell()
            reconstruct = True

        # Set degree
        degree = element.embedded_superdegree
        if degree is None:
            degree = common_degree
            reconstruct = True

        # Reconstruct element and add to map
        if reconstruct:
            element_mapping[element] = element.reconstruct(cell=cell, degree=degree)
        else:
            element_mapping[element] = element

    return element_mapping


def _compute_max_subdomain_ids(integral_data: list[IntegralData]) -> dict[str, int]:
    """Compute the maximum subdomain ids."""
    max_subdomain_ids = {}
    for itg_data in integral_data:
        it = itg_data.integral_type
        for integral in itg_data.integrals:
            # Convert string for default integral to -1
            sids = (-1 if isinstance(si, str) else si for si in integral.subdomain_id())
            newmax = max(sids) + 1
            prevmax = max_subdomain_ids.get(it, 0)
            max_subdomain_ids[it] = max(prevmax, newmax)
    return max_subdomain_ids


def _check_elements(form_data):
    """Check elements."""
    for element in chain(form_data.unique_elements, form_data.unique_sub_elements):
        if element.cell is None:
            raise ValueError(f"Found element with undefined cell: {element}")


def _check_facet_geometry(integral_data):
    """Check facet geometry."""
    for itg_data in integral_data:
        for itg in itg_data.integrals:
            for expr in traverse_unique_terminals(itg.integrand()):
                cls = expr._ufl_class_
                if issubclass(cls, GeometricFacetQuantity):
                    domain = extract_unique_domain(expr, expand_mesh_sequence=False)
                    if isinstance(domain, MeshSequence):
                        raise RuntimeError(
                            f"Not expecting a terminal object on a "
                            f"mesh sequence at this stage: found {expr!r}"
                        )
                    it = itg_data.domain_integral_type_map[domain]
                    # Facet geometry is only valid in facet integrals.
                    # Allowing custom integrals to pass as well, although
                    # that's not really strict enough.
                    if not ("facet" in it or "custom" in it or "interface" in it):
                        # Not a facet integral
                        raise ValueError(f"Integral of type {it} cannot contain a {cls.__name__}.")


def _build_coefficient_replace_map(
    coefficients: list[BaseCoefficient], element_mapping=None
) -> tuple[list[BaseCoefficient], dict[BaseCoefficient, BaseCoefficient]]:
    """Create new Coefficient objects with count starting at 0.

    Returns:
        lists of the new objects and mapping from old to new objects
    """
    if element_mapping is None:
        element_mapping = {}

    new_coefficients = []
    replace_map = {}
    for i, f in enumerate(coefficients):
        old_e = f.ufl_element()
        new_e = element_mapping.get(old_e, old_e)
        # XXX: This is a hack to ensure that if the original
        # coefficient had a domain, the new one does too.
        # This should be overhauled with requirement that Expressions
        # always have a domain.
        domain = extract_unique_domain(f, expand_mesh_sequence=False)
        if domain is not None:
            new_e = FunctionSpace(domain, new_e)
        new_f = Coefficient(new_e, count=i)
        new_coefficients.append(new_f)
        replace_map[f] = new_f

    return new_coefficients, replace_map


class FormData:
    """Class collecting various information extracted from a Form by calling preprocess."""

    _original_form: Form
    _integral_data: list[IntegralData]
    _reduced_coefficients: list[BaseCoefficient]
    _original_coefficient_positions: list[tuple[int, BaseCoefficient]]
    _function_replace_map: dict[BaseCoefficient, BaseCoefficient]
    _coefficient_elements: list[Any]
    _coefficient_split: dict[Coefficient, list[Coefficient]]

    def __init__(
        self,
        original_form: Form,
        integral_data: list[IntegralData],
        do_apply_default_restrictions: bool = True,
        do_apply_restrictions: bool = True,
        do_replace_functions: bool = False,
        coefficients_to_split: tuple[BaseCoefficient, ...] | None = None,
        complex_mode: bool = False,
    ):
        """Create form-data for a form that has been processed.

        Args:
            original_form: The form.
            integral_data: List of integral data objects corresponding to
                the form.
            do_apply_default_restrictions: Apply default restrictions, defined in
                {py:mod}`ufl.algorithms.apply_restrictions` to integrals if no
                restriction has been set.
            do_apply_restrictions: Apply restrictions towards terminal nodes.
            do_replace_functions: Replace functions with with its cannonically numbered
                function or thos provided in coefficients_to_split.
            coefficients_to_split: Sequence of coefficients to split over a MeshSequence
            complex_mode: If form has been processed as complex or not.
        """
        self._original_form = original_form
        self._integral_data = integral_data

        # --- Create replacements for arguments and coefficients

        # Figure out which form coefficients each integral should enable
        for itg_data in self.integral_data:
            itg_coeffs = set()
            # Get all coefficients in integrand
            for itg in itg_data.integrals:
                itg_coeffs.update(extract_coefficients(itg.integrand()))
            # Store with IntegralData object
            itg_data.integral_coefficients = itg_coeffs

        # Figure out which coefficients from the original form are
        # actually used in any integral (Differentiation may reduce the
        # set of coefficients w.r.t. the original form)
        reduced_coefficients_set = set()
        for itg_data in self.integral_data:
            reduced_coefficients_set.update(itg_data.integral_coefficients)
        self._reduced_coefficients = sorted(reduced_coefficients_set, key=lambda c: c.count())
        self._original_coefficient_positions = [
            i
            for i, c in enumerate(self.original_form.coefficients())
            if c in self.reduced_coefficients
        ]

        # Store back into integral data which form coefficients are used
        # by each integral
        for itg_data in self.integral_data:
            itg_data.enabled_coefficients = [
                bool(coeff in itg_data.integral_coefficients) for coeff in self.reduced_coefficients
            ]

        # Mappings from elements and coefficients that reside in form to
        # objects with canonical numbering as well as completed cells and
        # elements
        renumbered_coefficients, function_replace_map = _build_coefficient_replace_map(
            self.reduced_coefficients, self.element_replace_map
        )
        self._function_replace_map = function_replace_map

        self._coefficient_elements = tuple(f.ufl_element() for f in renumbered_coefficients)

        # --- Apply replace(integrand, self.function_replace_map)
        if do_replace_functions:
            for itg_data in self.integral_data:
                new_integrals = []
                for integral in itg_data.integrals:
                    integrand = replace(integral.integrand(), self.function_replace_map)
                    new_integrals.append(integral.reconstruct(integrand=integrand))
                itg_data.integrals = new_integrals

        # --- Split mixed coefficients with their components
        if coefficients_to_split is None:
            self._coefficient_split = {}
        else:
            if not do_replace_functions:
                raise ValueError("Must call with do_replace_functions=True")
            for itg_data in self.integral_data:
                new_integrals = []
                for integral in itg_data.integrals:
                    # Propagate restrictions as required by CoefficientSplitter.
                    # Can not yet apply default restrictions at this point
                    # as restrictions can only be applied to the components.
                    new_integral = apply_restrictions(integral)
                    new_integrals.append(new_integral)
                itg_data.integrals = new_integrals
            # Split coefficients that are contained in ``coefficients_to_split``
            # into components, and store a dict in ``self`` that maps
            # each coefficient to its components.
            coefficient_split = {}
            for o in self.reduced_coefficients:
                if o in coefficients_to_split:
                    c = self.function_replace_map[o]
                    mesh = extract_unique_domain(c, expand_mesh_sequence=False)
                    elem = c.ufl_element()
                    coefficient_split[c] = [
                        Coefficient(FunctionSpace(m, e))
                        for m, e in zip(mesh.iterable_like(elem), elem.sub_elements)
                    ]
            self._coefficient_split = coefficient_split
            coeff_splitter = CoefficientSplitter(self.coefficient_split)
            for itg_data in self.integral_data:
                new_integrals = []
                for integral in itg_data.integrals:
                    integrand = coeff_splitter(integral.integrand())
                    # Potentially need to call `remove_component_tensors()` here, but
                    # early-simplifications of Indexed objects seem sufficient.
                    if not isinstance(integrand, Zero):
                        new_integrals.append(integral.reconstruct(integrand=integrand))
                itg_data.integrals = new_integrals

        # Propagate restrictions to terminals
        if do_apply_restrictions:
            for itg_data in self.integral_data:
                # Need the following if block in case not all participating domains
                # have been included in the Measure (backwards compat).
                if all(
                    not integral_type.startswith("interior_facet")
                    for _, integral_type in itg_data.domain_integral_type_map.items()
                ):
                    continue
                if do_apply_default_restrictions:
                    default_restrictions = {
                        domain: default_restriction_map[integral_type]
                        for domain, integral_type in itg_data.domain_integral_type_map.items()
                    }
                    # Need the following dict update in case not all participating domains
                    # have been included in the Measure (backwards compat).
                    extra = {
                        domain: default_restriction_map[itg_data.integral_type]
                        for integral in itg_data.integrals
                        for domain in extract_domains(integral)
                        if domain not in default_restrictions
                    }
                    default_restrictions.update(extra)
                else:
                    default_restrictions = None
                new_integrals = []
                for integral in itg_data.integrals:
                    new_integral = apply_restrictions(
                        integral,
                        default_restrictions=default_restrictions,
                    )
                    new_integrals.append(new_integral)
                itg_data.integrals = new_integrals

        _check_elements(self)
        _check_facet_geometry(self.integral_data)
        _check_form_arity(self.integral_data, self.original_form.arguments(), complex_mode)

    def __str__(self):
        """Return formatted summary of form data."""
        types = sorted(self.max_subdomain_ids.keys())
        geometry = (("Geometric dimension", self.geometric_dimension),)
        subdomains = tuple(
            (f"Number of {integral_type} subdomains", self.max_subdomain_ids[integral_type])
            for integral_type in types
        )
        functions = (
            # Arguments
            ("Rank", self.rank),
            ("Arguments", lstr(self.original_form.arguments())),
            # Coefficients
            ("Number of coefficients", self.num_coefficients),
            ("Coefficients", lstr(self.reduced_coefficients)),
            # Elements
            ("Unique elements", estr(self.unique_elements)),
            ("Unique sub elements", estr(self.unique_sub_elements)),
        )
        return tstr(geometry + subdomains + functions)

    @property
    def integral_data(self) -> list[IntegralData]:
        """The integral data of the form."""
        return self._integral_data

    @property
    def reduced_coefficients(self) -> list[Coefficient]:
        """Set of active coeffcient in the form."""
        return self._reduced_coefficients

    @property
    def num_coefficients(self):
        """Number of active coefficients in the form."""
        return len(self.reduced_coefficients)

    @property
    def rank(self):
        """Rank of the form."""
        return len(self.original_form.arguments())

    @property
    def geometric_dimension(self):
        """Geometric dimension of the form."""
        return self.original_form.integrals()[0].ufl_domain().geometric_dimension

    @property
    def function_replace_map(self) -> dict[Coefficient, Coefficient]:
        """Map from coefficients in form to those used in IntegralData."""
        return self._function_replace_map

    @cached_property
    def element_replace_map(self):
        """Mapping from incomplete elements to new well-defined elements.

        This is to support the Expression construct in dolfin which
        subclasses Coefficient but doesn't provide an element,
        and the Constant construct that doesn't provide the domain that
        a Coefficient is supposed to have. A future design iteration in
        UFL/UFC/FFC/DOLFIN may allow removal of this mapping with the
        introduction of UFL types for Expression-like functions that can
        be evaluated in quadrature points.

        Note:
            This property should likely be removed.
        """
        return _compute_element_mapping(self.original_form)

    @cached_property
    def max_subdomain_ids(self) -> dict[str, int]:
        """For each integral type, return the maximum subdomain id."""
        return _compute_max_subdomain_ids(self.integral_data)

    @cached_property
    def argument_elements(self) -> list[Any]:
        """The set of elements the arguments in the form."""
        return tuple(f.ufl_element() for f in self.original_form.arguments())

    @property
    def coefficient_elements(self) -> list[Any]:
        """The set of elements used for coefficients in the form."""
        return self._coefficient_elements

    @cached_property
    def coordinate_elements(self):
        """The set of coordinate elements in the form."""
        return tuple(domain.ufl_coordinate_element() for domain in self.original_form.ufl_domains())

    @cached_property
    def unique_elements(self):
        """Set of unique elements (not expanded for sub elements) in the form."""
        return unique_tuple(
            self.argument_elements + self.coefficient_elements + self.coordinate_elements
        )

    @cached_property
    def unique_sub_elements(self):
        """Set of unique elements (expanded by sub elements) in the form."""
        all_elements = self.argument_elements + self.coefficient_elements + self.coordinate_elements
        all_sub_elements = extract_sub_elements(all_elements)
        return unique_tuple(all_sub_elements)

    @property
    def coefficient_split(self) -> dict[Coefficient, list[Coefficient]]:
        """Map from coefficient to its split counterparts."""
        return self._coefficient_split

    @cached_property
    def preprocessed_form(self):
        """This is used in tests and is rather slow."""
        return reconstruct_form_from_integral_data(self.integral_data)

    @property
    def original_form(self):
        """The input UFL form."""
        return self._original_form

    @property
    def original_coefficient_positions(self):
        """Original placement of coefficients in form."""
        return self._original_coefficient_positions
