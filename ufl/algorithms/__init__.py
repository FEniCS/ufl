"This module collects algorithms and utility functions operating on UFL objects."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-14 -- 2009-05-14"

# Modified by Anders Logg, 2008-2009.

# Function for preprocessing a form
from ufl.algorithms.preprocess import preprocess

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
                                    extract_variables, extract_duplications, \
                                    extract_max_quadrature_element_degree, estimate_quadrature_degree, \
                                    sort_elements

# Utilities for checking properties of forms
from ufl.algorithms.predicates import is_multilinear

# Utilities for error checking of forms
from ufl.algorithms.checks import validate_form

# Utilites for modifying expressions and forms
from ufl.algorithms.transformations import transform, Transformer, ReuseTransformer, apply_transformer, \
                                           ufl2ufl, ufl2uflcopy, \
                                           replace, flatten, strip_variables, \
                                           expand_compounds, \
                                           mark_duplications, \
                                           estimate_max_polynomial_degree, estimate_total_polynomial_degree
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.expand_indices import expand_indices, expand_indices2, purge_list_tensors
from ufl.algorithms.propagate_restrictions import propagate_restrictions

# Utilities for transforming complete Forms into other Forms
from ufl.algorithms.formtransformations import compute_form_adjoint, compute_form_action, compute_energy_norm, \
                                               compute_form_lhs, compute_form_rhs, compute_form_functional #, compute_dirichlet_functional

# Utilities for Automatic Functional Differentiation
from ufl.algorithms.ad import expand_derivatives #, compute_diff, propagate_spatial_derivatives, compute_form_derivative

# Utilities for working with linearized computational graphs
from ufl.algorithms.graph import Graph, format_graph, rebuild_tree, partition # TODO: Add more imports here

# Utilities for tuple notation
from ufl.algorithms.tuplenotation import tuple2form, as_form

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
