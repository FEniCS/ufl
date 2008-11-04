"This module collects algorithms and utility functions operating on UFL objects."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-14 -- 2008-10-30"

# Modified by Anders Logg, 2008

# Utilities for traversing over expression trees in different ways
from .traversal import iter_expressions
from .traversal import post_traversal, pre_traversal, traversal
from .traversal import post_walk, pre_walk, walk

# Utilities for extracting information from forms and expressions
from .analysis import extract_type, extract_classes, extract_domain, extract_value_shape
from .analysis import extract_basisfunctions, extract_coefficients, extract_elements, extract_unique_elements
from .analysis import extract_variables
from .analysis import extract_monomials

# Utility class for easy collecting of data about form
from .formdata import FormData

# Utilities for checking properties of forms
from .predicates import is_multilinear

# Utilities for error checking of forms
from .checks import validate_form

# Utilites for modifying expressions and forms
from .transformations import transform, transform_integrands
from .transformations import ufl2ufl, ufl2uflcopy
from .transformations import expand_compounds, flatten
from .transformations import replace, replace_in_form

# Utilities for working with indices
from .indexalgorithms import renumber_indices, substitute_indices, expand_indices

# Utilities for working with variables
from .variables import strip_variables, extract_variables, extract_duplications, mark_duplications

# Utilities for working with dependencies of subexpressions
from .dependencies import split_by_dependencies

# Utilities for transforming complete Forms into other Forms
from .formtransformations import compute_form_adjoint, compute_form_action
from .formtransformations import compute_form_lhs, compute_form_rhs
#from .formtransformations import compute_dirichlet_functional

# Utilities for Automatic Functional Differentiation
from .ad import compute_form_derivative, compute_diff, propagate_spatial_derivatives

# Utilities for UFL object printing
from .ufl2latex import ufl2latex, ufl2tex, ufl2pdf
from .printing import integral_info, form_info, tree_format

# Utilities for form file handling
from .formfiles import load_forms
