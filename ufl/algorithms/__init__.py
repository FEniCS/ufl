"This module collects algorithms and utility functions operating on UFL objects."

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
# Modified by Anders Logg, 2008-2009.
#
# First added:  2008-08-14
# Last changed: 2011-10-11

# Function for preprocessing a form
from ufl.algorithms.preprocess import preprocess, preprocess_expression, extract_common_cell

# Class for simple extraction of form meta data
from ufl.algorithms.formdata import FormData

# Utilities for traversing over expression trees in different ways
from ufl.algorithms.traversal import iter_expressions, traverse_terminals, \
                                     post_traversal, pre_traversal, \
                                     post_walk, pre_walk, walk

# Utilities for extracting information from forms and expressions
from ufl.algorithms.analysis import extract_classes, extract_type, has_type, \
                                    extract_arguments, extract_coefficients, extract_arguments_and_coefficients, \
                                    extract_elements, extract_unique_elements, \
                                    extract_sub_elements, extract_unique_sub_elements, \
                                    extract_variables, extract_duplications, \
                                    extract_max_quadrature_element_degree, estimate_quadrature_degree, \
                                    sort_elements

# Utilities for checking properties of forms
from ufl.algorithms.predicates import is_multilinear

# Utilities for error checking of forms
from ufl.algorithms.checks import validate_form

# Utilites for modifying expressions and forms
from ufl.algorithms.multifunction import MultiFunction
from ufl.algorithms.transformer import Transformer, is_post_handler, \
                                       transform, transform_integrands, apply_transformer, \
                                       ReuseTransformer, ufl2ufl, \
                                       CopyTransformer, ufl2uflcopy, \
                                       VariableStripper, strip_variables
from ufl.algorithms.replace import Replacer, replace
from ufl.algorithms.expand_compounds import CompoundExpander, expand_compounds, \
                                            CompoundExpanderPreDiff, expand_compounds_prediff, \
                                            CompoundExpanderPostDiff, expand_compounds_postdiff
from ufl.algorithms.estimate_degrees import SumDegreeEstimator, estimate_total_polynomial_degree
from ufl.algorithms.argument_dependencies import ArgumentDependencyExtracter, extract_argument_dependencies, NotMultiLinearException
from ufl.algorithms.deprecated import TreeFlattener, flatten, \
                                      DuplicationMarker, mark_duplications, \
                                      DuplicationPurger, purge_duplications
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.expand_indices import expand_indices, purge_list_tensors
from ufl.algorithms.propagate_restrictions import propagate_restrictions

# Utilities for transforming complete Forms into other Forms
from ufl.algorithms.formtransformations import compute_form_adjoint, compute_form_action, compute_energy_norm, \
                                               compute_form_lhs, compute_form_rhs, compute_form_functional, \
                                               compute_form_arities

# Utilities for Automatic Functional Differentiation
from ufl.algorithms.ad import expand_derivatives #, compute_diff, propagate_spatial_derivatives, compute_form_derivative

# Utilities for working with linearized computational graphs
from ufl.algorithms.graph import Graph, format_graph, rebuild_tree, partition # TODO: Add more imports here

# Utilities for tuple notation
from ufl.algorithms.tuplenotation import as_form

# Utilities for UFL object printing
from ufl.algorithms.printing import integral_info, form_info, tree_format
from ufl.algorithms.ufl2latex import ufl2latex, ufl2tex, ufl2pdf, forms2latexdocument
from ufl.algorithms.ufl2dot import ufl2dot

# Utilities for form file handling
from ufl.algorithms.formfiles import read_ufl_file, load_ufl_file, load_forms

# State of files (in the opinion of Martin):
#    traversal.py           - Ok.
#    analysis.py            - Ok, some unused stuff.
#    formdata.py            - Probably don't need both self.unique_elements and self.sub_elements?
#                             Need to improve quadrature order estimation.
#    graph.py               - Work in progress, works ok so far.
#    predicates.py          - is_multilinear seems ok but needs testing.
#    checks.py              - Ok, more checks are welcome.
#    formfiles.py           - Ok.
#    transformations.py     - Ok.
#    formtransformations.py - Ok? Needs testing.
#    ad.py                  - Ok?
#    printing.py            - Ok.
#    latextools.py          - Ok.
#    ufl2latex.py           - Fix precedence stuff.
#    ufl2dot.py             - Rework with graph tools.
