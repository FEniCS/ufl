# -*- coding: utf-8 -*-
# flake8: noqa
"This module collects algorithms and utility functions operating on UFL objects."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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
# Modified by Anders Logg, 2008-2009.


# FIXME: Clean up this to become a more official set of supported
#        algorithms.  Currently contains too much stuff that's not
#        recommended to use.


# Utilities for traversing over expression trees in different ways
from ufl.algorithms.traversal import iter_expressions

# Keeping these imports here for backwards compatibility, doesn't cost
# anything.  Prefer importing from ufl.corealg.traversal in future
# code.
from ufl.corealg.traversal import pre_traversal, post_traversal
from ufl.corealg.traversal import traverse_terminals, traverse_unique_terminals


# Utilities for extracting information from forms and expressions
from ufl.algorithms.analysis import (
    extract_type,
    extract_arguments,
    extract_coefficients,
    extract_arguments_and_coefficients,
    extract_elements,
    extract_unique_elements,
    extract_sub_elements,
    sort_elements,
    )


# Preprocessing a form to extract various meta data
from ufl.algorithms.formdata import FormData
from ufl.algorithms.compute_form_data import compute_form_data

# Utilities for checking properties of forms
from ufl.algorithms.predicates import is_multilinear
from ufl.algorithms.signature import compute_form_signature

# Utilities for error checking of forms
from ufl.algorithms.checks import validate_form

# Utilites for modifying expressions and forms
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.transformer import Transformer, is_post_handler, \
                                       apply_transformer, \
                                       ReuseTransformer, \
                                       VariableStripper, strip_variables
from ufl.algorithms.replace import Replacer, replace
from ufl.algorithms.change_to_reference import change_to_reference_grad
from ufl.algorithms.expand_compounds import expand_compounds
from ufl.algorithms.estimate_degrees import SumDegreeEstimator, estimate_total_polynomial_degree
from ufl.algorithms.argument_dependencies import ArgumentDependencyExtracter, extract_argument_dependencies, NotMultiLinearException
from ufl.algorithms.expand_indices import expand_indices, purge_list_tensors

# Utilities for transforming complete Forms into other Forms
from ufl.algorithms.formtransformations import (
    compute_form_adjoint, compute_form_action, compute_energy_norm,
    compute_form_lhs, compute_form_rhs,
    compute_form_functional, compute_form_arities)

from ufl.algorithms.formsplitter import FormSplitter

# Utilities for Automatic Functional Differentiation
from ufl.algorithms.ad import expand_derivatives

# Utilities for form file handling
from ufl.algorithms.formfiles import read_ufl_file, load_ufl_file, load_forms


# Utilities for UFL object printing
from ufl.formatting.printing import integral_info, form_info, tree_format
from ufl.formatting.ufl2latex import ufl2latex, ufl2tex, ufl2pdf, forms2latexdocument
from ufl.formatting.ufl2dot import ufl2dot
