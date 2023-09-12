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

# Utilities for Automatic Functional Differentiation
# Utilities for Automatic Functional Differentiation
from ufl.algorithms.ad import expand_derivatives
# Utilities for extracting information from forms and expressions
# Utilities for extracting information from forms and expressions
from ufl.algorithms.analysis import (  # extract_arguments_and_coefficients,
    extract_arguments, extract_coefficients, extract_elements,
    extract_sub_elements, extract_type, extract_unique_elements, sort_elements)
from ufl.algorithms.change_to_reference import change_to_reference_grad
# Utilities for error checking of forms
# Utilities for error checking of forms
from ufl.algorithms.checks import validate_form
# Preprocessing a form to extract various meta data
# Preprocessing a form to extract various meta data
# from ufl.algorithms.formdata import FormData
from ufl.algorithms.compute_form_data import compute_form_data
# from ufl.algorithms.estimate_degrees import SumDegreeEstimator
# from ufl.algorithms.estimate_degrees import SumDegreeEstimator
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree
from ufl.algorithms.expand_compounds import expand_compounds
from ufl.algorithms.expand_indices import expand_indices
# Utilities for form file handling
# Utilities for form file handling
from ufl.algorithms.formfiles import load_forms, load_ufl_file, read_ufl_file
from ufl.algorithms.formsplitter import FormSplitter
# Utilities for transforming complete Forms into other Forms
# Utilities for transforming complete Forms into other Forms
from ufl.algorithms.formtransformations import (compute_energy_norm,
                                                compute_form_action,
                                                compute_form_adjoint,
                                                compute_form_arities,
                                                compute_form_functional,
                                                compute_form_lhs,
                                                compute_form_rhs)
# from ufl.algorithms.replace import Replacer
# from ufl.algorithms.replace import Replacer
from ufl.algorithms.replace import replace
# Utilities for checking properties of forms
# Utilities for checking properties of forms
from ufl.algorithms.signature import compute_form_signature
from ufl.algorithms.strip_terminal_data import (replace_terminal_data,
                                                strip_terminal_data)
# from ufl.algorithms.transformer import is_post_handler
from ufl.algorithms.transformer import (ReuseTransformer, Transformer,
                                        apply_transformer, strip_variables)
# Utilites for modifying expressions and forms
# Utilites for modifying expressions and forms
from ufl.corealg.multifunction import MultiFunction
# Keeping these imports here for backwards compatibility, doesn't cost
# anything.  Prefer importing from ufl.corealg.traversal in future
# code.
from ufl.corealg.traversal import post_traversal
from ufl.utils.formatting import tree_format
