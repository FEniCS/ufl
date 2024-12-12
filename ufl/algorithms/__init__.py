"""This module collects algorithms and utility functions operating on UFL objects."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008-2009.


# FIXME: Clean up this to become a more official set of supported
#        algorithms.  Currently contains too much stuff that's not
#        recommended to use. The __all__ list below is a start based
#        on grepping of other FEniCS code for ufl.algorithm imports.

__all__ = [
    "FormSplitter",
    "MultiFunction",
    "ReuseTransformer",
    "Transformer",
    "apply_transformer",
    "change_to_reference_grad",
    "compute_energy_norm",
    "compute_form_action",
    "compute_form_adjoint",
    "compute_form_arities",
    "compute_form_data",
    "compute_form_functional",
    "compute_form_lhs",
    "compute_form_rhs",
    "compute_form_signature",
    "estimate_total_polynomial_degree",
    "expand_derivatives",
    "expand_indices",
    "extract_arguments",
    "extract_base_form_operators",
    "extract_coefficients",
    "extract_elements",
    "extract_sub_elements",
    "extract_type",
    "extract_unique_elements",
    "load_forms",
    "load_ufl_file",
    "post_traversal",
    "preprocess_form",
    "read_ufl_file",
    "replace",
    "replace_terminal_data",
    "sort_elements",
    "strip_terminal_data",
    "strip_variables",
    "tree_format",
    "validate_form",
]

from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.analysis import (
    extract_arguments,
    extract_base_form_operators,
    extract_coefficients,
    extract_elements,
    extract_sub_elements,
    extract_type,
    extract_unique_elements,
    sort_elements,
)
from ufl.algorithms.change_to_reference import change_to_reference_grad
from ufl.algorithms.checks import validate_form
from ufl.algorithms.compute_form_data import compute_form_data, preprocess_form
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.algorithms.expand_indices import expand_indices
from ufl.algorithms.formfiles import load_forms, load_ufl_file, read_ufl_file
from ufl.algorithms.formsplitter import FormSplitter
from ufl.algorithms.formtransformations import (
    compute_energy_norm,
    compute_form_action,
    compute_form_adjoint,
    compute_form_arities,
    compute_form_functional,
    compute_form_lhs,
    compute_form_rhs,
)
from ufl.algorithms.replace import replace
from ufl.algorithms.signature import compute_form_signature
from ufl.algorithms.strip_terminal_data import replace_terminal_data, strip_terminal_data
from ufl.algorithms.transformer import (
    ReuseTransformer,
    Transformer,
    apply_transformer,
    strip_variables,
)
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.traversal import post_traversal
from ufl.utils.formatting import tree_format
