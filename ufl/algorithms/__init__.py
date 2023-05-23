# -*- coding: utf-8 -*-
# flake8: noqa
"This module collects algorithms and utility functions operating on UFL objects."

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
    "estimate_total_polynomial_degree",
    "sort_elements",
    "compute_form_data",
    "purge_list_tensors",
    "apply_transformer",
    "ReuseTransformer",
    "load_ufl_file",
    "Transformer",
    "MultiFunction",
    "extract_unique_elements",
    "extract_type",
    "extract_elements",
    "extract_sub_elements",
    "expand_indices",
    "replace",
    "expand_derivatives",
    "extract_coefficients",
    "strip_variables",
    "strip_terminal_data",
    "replace_terminal_data",
    "post_traversal",
    "change_to_reference_grad",
    "expand_compounds",
    "validate_form",
    "FormSplitter",
    "extract_arguments",
    "compute_form_adjoint",
    "compute_form_action",
    "compute_energy_norm",
    "compute_form_lhs",
    "compute_form_rhs",
    "compute_form_functional",
    "compute_form_signature",
    "tree_format",
]

# Utilities for traversing over expression trees in different ways
# from ufl.algorithms.traversal import iter_expressions

# Keeping these imports here for backwards compatibility, doesn't cost
# anything.  Prefer importing from ufl.corealg.traversal in future
# code.
# from ufl.corealg.traversal import pre_traversal
from ufl.corealg.traversal import post_traversal
# from ufl.corealg.traversal import traverse_terminals, traverse_unique_terminals


# Utilities for extracting information from forms and expressions
from ufl.algorithms.analysis import (
    extract_type,
    extract_arguments,
    extract_coefficients,
    # extract_arguments_and_coefficients,
    extract_elements,
    extract_unique_elements,
    extract_sub_elements,
    sort_elements,
)


# Preprocessing a form to extract various meta data
# from ufl.algorithms.formdata import FormData
from ufl.algorithms.compute_form_data import compute_form_data

# Utilities for checking properties of forms
from ufl.algorithms.signature import compute_form_signature

# Utilities for error checking of forms
from ufl.algorithms.checks import validate_form

# Utilites for modifying expressions and forms
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.transformer import Transformer, ReuseTransformer
# from ufl.algorithms.transformer import is_post_handler
from ufl.algorithms.transformer import apply_transformer
from ufl.algorithms.transformer import strip_variables
from ufl.algorithms.strip_terminal_data import strip_terminal_data
from ufl.algorithms.strip_terminal_data import replace_terminal_data
# from ufl.algorithms.replace import Replacer
from ufl.algorithms.replace import replace
from ufl.algorithms.change_to_reference import change_to_reference_grad
from ufl.algorithms.expand_compounds import expand_compounds
# from ufl.algorithms.estimate_degrees import SumDegreeEstimator
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.algorithms.expand_indices import expand_indices, purge_list_tensors

# Utilities for transforming complete Forms into other Forms
from ufl.algorithms.formtransformations import compute_form_adjoint
from ufl.algorithms.formtransformations import compute_form_action
from ufl.algorithms.formtransformations import compute_energy_norm
from ufl.algorithms.formtransformations import compute_form_lhs
from ufl.algorithms.formtransformations import compute_form_rhs
from ufl.algorithms.formtransformations import compute_form_functional
from ufl.algorithms.formtransformations import compute_form_arities

from ufl.algorithms.formsplitter import FormSplitter

# Utilities for Automatic Functional Differentiation
from ufl.algorithms.ad import expand_derivatives

# Utilities for form file handling
from ufl.algorithms.formfiles import read_ufl_file
from ufl.algorithms.formfiles import load_ufl_file
from ufl.algorithms.formfiles import load_forms

from ufl.utils.formatting import tree_format
