"This module collects algorithms and utility functions operating on UFL objects."


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-14 -- 2008-11-17"

# Modified by Anders Logg, 2008

# Utilities for traversing over expression trees in different ways
from ufl.algorithms.traversal import iter_expressions
from ufl.algorithms.traversal import post_traversal, pre_traversal, traversal, traverse_terminals
from ufl.algorithms.traversal import post_walk, pre_walk, walk

# Utilities for extracting information from forms and expressions
from ufl.algorithms.analysis import extract_type, extract_classes
from ufl.algorithms.analysis import extract_basisfunctions, extract_coefficients, extract_elements, extract_unique_elements
from ufl.algorithms.analysis import extract_variables
from ufl.algorithms.analysis import extract_monomials

# Utility class for easy collecting of data about form
from ufl.algorithms.formdata import FormData

# Utilities for checking properties of forms
from ufl.algorithms.predicates import is_multilinear

# Utilities for error checking of forms
from ufl.algorithms.checks import validate_form

# Utilites for modifying expressions and forms
from ufl.algorithms.transformations import transform, transform_integrands
from ufl.algorithms.transformations import ufl2ufl, ufl2uflcopy
from ufl.algorithms.transformations import expand_compounds, flatten
from ufl.algorithms.transformations import replace, replace_in_form

# Utilities for working with indices
from ufl.algorithms.indexalgorithms import renumber_indices, substitute_indices, expand_indices

# Utilities for working with variables
from ufl.algorithms.variables import strip_variables, extract_variables, extract_duplications, mark_duplications

# Utilities for working with dependencies of subexpressions
from ufl.algorithms.dependencies import split_by_dependencies

# Utilities for transforming complete Forms into other Forms
from ufl.algorithms.formtransformations import compute_form_adjoint, compute_form_action
from ufl.algorithms.formtransformations import compute_form_lhs, compute_form_rhs
#from ufl.algorithms.formtransformations import compute_dirichlet_functional

# Utilities for Automatic Functional Differentiation
from ufl.algorithms.ad import compute_form_derivative, compute_diff, propagate_spatial_derivatives

# Utilities for UFL object printing
from ufl.algorithms.ufl2latex import ufl2latex, ufl2tex, ufl2pdf
from ufl.algorithms.printing import integral_info, form_info, tree_format
from ufl.algorithms.ufl2dot import ufl2dot

# Utilities for form file handling
from ufl.algorithms.formfiles import load_forms
