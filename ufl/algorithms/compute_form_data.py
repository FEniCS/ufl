"""This module provides the compute_form_data function.

Form compilers will typically call compute_form_dataprior to code
generation to preprocess/simplify a raw input form given by a user.
"""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering

# These are the main symbolic processing steps:
from ufl.algorithms.apply_derivatives import apply_coordinate_derivatives, apply_derivatives
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_integral_scaling import apply_integral_scaling
from ufl.algorithms.comparison_checker import do_comparison_check

# See TODOs at the call sites of these below:
from ufl.algorithms.domain_analysis import (
    build_integral_data,
    group_form_integrals,
)
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.algorithms.formdata import FormData
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms.remove_component_tensors import remove_component_tensors
from ufl.classes import Form


def attach_estimated_degrees(form):
    """Attach estimated polynomial degree to a form's integrals.

    Args:
        form: The Form` to inspect.

    Returns:
        A new Form with estimate degrees attached.
    """
    integrals = form.integrals()

    new_integrals = []
    for integral in integrals:
        md = {}
        md.update(integral.metadata())
        degree = estimate_total_polynomial_degree(integral.integrand())
        md["estimated_polynomial_degree"] = degree
        new_integrals.append(integral.reconstruct(metadata=md))
    return Form(new_integrals)


def preprocess_form(form, complex_mode):
    """Preprocess a form."""
    # Note: Default behaviour here will process form the way that is
    # currently expected by vanilla FFC

    # Check that the form does not try to compare complex quantities:
    # if the quantites being compared are 'provably' real, wrap them
    # with Real, otherwise throw an error.
    if complex_mode:
        form = do_comparison_check(form)

    # Lower abstractions for tensor-algebra types into index notation,
    # reducing the number of operators later algorithms and form
    # compilers need to handle
    form = apply_algebra_lowering(form)

    # After lowering to index notation, remove any complex nodes that
    # have been introduced but are not wanted when working in real mode,
    # allowing for purely real forms to be written
    if not complex_mode:
        form = remove_complex_nodes(form)

    # Apply differentiation before function pullbacks, because for
    # example coefficient derivatives are more complicated to derive
    # after coefficients are rewritten, and in particular for
    # user-defined coefficient relations it just gets too messy
    form = apply_derivatives(form)

    return form


def compute_form_data(
    form,
    do_apply_function_pullbacks=False,
    do_apply_integral_scaling=False,
    do_apply_geometry_lowering=False,
    preserve_geometry_types=(),
    do_apply_default_restrictions=True,
    do_apply_restrictions=True,
    do_estimate_degrees=True,
    do_append_everywhere_integrals=True,
    do_replace_functions=False,
    coefficients_to_split=None,
    complex_mode=False,
    do_remove_component_tensors=False,
) -> FormData:
    """Compute form data.

    Args:
        form: The form to compute form data for.
        do_apply_function_pullbacks: Apply pull-back to reference cell
            for coefficients, including Piola and symmetry transforms
            if required.
        do_apply_integral_scaling: Apply scaling of moving the integral
            from physical to reference frame.
        do_apply_geometry_lowering: Lower the representation of geometrical
            quantities to a smaller subset of quantities
        preserve_geometry_types: Set of quantities not to lower, and keep
            at its present stage for the form-compiler.
        do_apply_default_restrictions: Apply default restrictions, defined in
            {py:mod}`ufl.algorithms.apply_restrictions` to integrals if no
            restriction has been set.
        do_apply_restrictions: Apply restrictions towards terminal nodes.
        do_replace_functions: Replace functions with with its cannonically numbered
            function or thos provided in coefficients_to_split.
        coefficients_to_split: Sequence of coefficients to split over a MeshSequence.
        do_estimate_degrees: Estimate polynomial degree of integrands.
        do_append_everywhere_integrals: If True append every `dx` integral to each `dx(i)`
            integral defined in the form.
        do_remove_component_tensors: Remove component-tensor if true.
        complex_mode: If false remove complex nodes from the form.
    """
    # --- Store untouched form for reference.
    # The user of FormData may get original arguments,
    # original coefficients, and form signature from this object.
    # But be aware that the set of original coefficients are not
    # the same as the ones used in the final UFC form.
    # See 'reduced_coefficients' below.
    original_form = form

    # --- Pass form integrands through some symbolic manipulation

    form = preprocess_form(form, complex_mode)

    # --- Group form integrals
    # TODO: Refactor this, it's rather opaque what this does
    # TODO: Is self.original_form.ufl_domains() right here?
    #       It will matter when we start including 'num_domains' in ufc form.
    form = group_form_integrals(
        form,
        original_form.ufl_domains(),
        do_append_everywhere_integrals=do_append_everywhere_integrals,
    )

    # Estimate polynomial degree of integrands now, before applying
    # any pullbacks and geometric lowering.  Otherwise quad degrees
    # blow up horrifically.
    if do_estimate_degrees:
        form = attach_estimated_degrees(form)

    if do_apply_function_pullbacks:
        # Rewrite coefficients and arguments in terms of their
        # reference cell values with Piola transforms and symmetry
        # transforms injected where needed.
        # Decision: Not supporting grad(dolfin.Expression) without a
        #           Domain.  Current dolfin works if Expression has a
        #           cell but this should be changed to a mesh.
        form = apply_function_pullbacks(form)

    # Scale integrals to reference cell frames
    if do_apply_integral_scaling:
        form = apply_integral_scaling(form)

    # Lower abstractions for geometric quantities into a smaller set
    # of quantities, allowing the form compiler to deal with a smaller
    # set of types and treating geometric quantities like any other
    # expressions w.r.t. loop-invariant code motion etc.
    if do_apply_geometry_lowering:
        form = apply_geometry_lowering(form, preserve_geometry_types)

    # Apply differentiation again, because the algorithms above can
    # generate new derivatives or rewrite expressions inside
    # derivatives
    if do_apply_function_pullbacks or do_apply_geometry_lowering:
        form = apply_derivatives(form)

        # Neverending story: apply_derivatives introduces new Jinvs,
        # which needs more geometry lowering
        if do_apply_geometry_lowering:
            form = apply_geometry_lowering(form, preserve_geometry_types)
            # Lower derivatives that may have appeared
            form = apply_derivatives(form)

    form = apply_coordinate_derivatives(form)

    # If in real mode, remove any complex nodes introduced during form processing.
    if not complex_mode:
        form = remove_complex_nodes(form)

    # Remove component tensors
    if do_remove_component_tensors:
        form = remove_component_tensors(form)
    integral_data = build_integral_data(form.integrals())
    return FormData(
        original_form,
        integral_data,
        do_apply_default_restrictions=do_apply_default_restrictions,
        do_apply_restrictions=do_apply_restrictions,
        do_replace_functions=do_replace_functions,
        coefficients_to_split=coefficients_to_split,
        complex_mode=complex_mode,
    )
